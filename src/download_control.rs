use std::sync::{
    Condvar, Mutex,
    atomic::{AtomicU8, Ordering},
};

pub const DOWNLOAD_CANCELLED_ERROR: &str =
    "模型下载已取消；未完成文件已保留，下次可以从已有进度继续。";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DownloadState {
    Idle = 0,
    Running = 1,
    Paused = 2,
    Cancelled = 3,
}

pub struct DownloadControl {
    state: AtomicU8,
    wait_lock: Mutex<()>,
    wake: Condvar,
    process_id: Mutex<Option<u32>>,
}

impl DownloadControl {
    const fn new() -> Self {
        Self {
            state: AtomicU8::new(DownloadState::Idle as u8),
            wait_lock: Mutex::new(()),
            wake: Condvar::new(),
            process_id: Mutex::new(None),
        }
    }

    pub fn begin(&self) -> DownloadSession<'_> {
        if let Ok(mut process_id) = self.process_id.lock() {
            *process_id = None;
        }
        self.state
            .store(DownloadState::Running as u8, Ordering::Release);
        DownloadSession { control: self }
    }

    pub fn state(&self) -> DownloadState {
        match self.state.load(Ordering::Acquire) {
            1 => DownloadState::Running,
            2 => DownloadState::Paused,
            3 => DownloadState::Cancelled,
            _ => DownloadState::Idle,
        }
    }

    pub fn pause(&self) -> bool {
        if self
            .state
            .compare_exchange(
                DownloadState::Running as u8,
                DownloadState::Paused as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return false;
        }
        self.signal_registered_process(ProcessSignal::Pause);
        true
    }

    pub fn resume(&self) -> bool {
        if self
            .state
            .compare_exchange(
                DownloadState::Paused as u8,
                DownloadState::Running as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return false;
        }
        self.signal_registered_process(ProcessSignal::Resume);
        self.wake.notify_all();
        true
    }

    pub fn cancel(&self) -> bool {
        loop {
            let current = self.state();
            if !matches!(current, DownloadState::Running | DownloadState::Paused) {
                return false;
            }
            if self
                .state
                .compare_exchange(
                    current as u8,
                    DownloadState::Cancelled as u8,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                break;
            }
        }
        self.signal_registered_process(ProcessSignal::Cancel);
        self.wake.notify_all();
        true
    }

    /// Called between streaming reads. A paused Rust download sleeps on a
    /// condition variable, so pausing does not introduce a polling loop.
    pub fn checkpoint(&self) -> Result<(), String> {
        let mut guard = self
            .wait_lock
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        loop {
            match self.state() {
                DownloadState::Cancelled => return Err(DOWNLOAD_CANCELLED_ERROR.to_owned()),
                DownloadState::Paused => {
                    guard = self
                        .wake
                        .wait(guard)
                        .unwrap_or_else(|error| error.into_inner());
                }
                DownloadState::Idle | DownloadState::Running => return Ok(()),
            }
        }
    }

    pub fn register_process(&self, process_id: u32) {
        if let Ok(mut current) = self.process_id.lock() {
            *current = Some(process_id);
        }
        match self.state() {
            DownloadState::Paused => signal_process(process_id, ProcessSignal::Pause),
            DownloadState::Cancelled => signal_process(process_id, ProcessSignal::Cancel),
            DownloadState::Idle | DownloadState::Running => {}
        }
    }

    pub fn clear_process(&self, process_id: u32) {
        if let Ok(mut current) = self.process_id.lock()
            && *current == Some(process_id)
        {
            *current = None;
        }
    }

    fn signal_registered_process(&self, signal: ProcessSignal) {
        let process_id = self.process_id.lock().ok().and_then(|current| *current);
        if let Some(process_id) = process_id {
            signal_process(process_id, signal);
        }
    }

    fn finish(&self) {
        if let Ok(mut process_id) = self.process_id.lock() {
            *process_id = None;
        }
        self.state
            .store(DownloadState::Idle as u8, Ordering::Release);
        self.wake.notify_all();
    }
}

pub struct DownloadSession<'a> {
    control: &'a DownloadControl,
}

impl Drop for DownloadSession<'_> {
    fn drop(&mut self) {
        self.control.finish();
    }
}

#[derive(Clone, Copy)]
enum ProcessSignal {
    Pause,
    Resume,
    Cancel,
}

#[cfg(unix)]
fn signal_process(process_id: u32, signal: ProcessSignal) {
    let Ok(process_id) = i32::try_from(process_id) else {
        return;
    };
    // Controlled commands are started as process-group leaders. Signalling the
    // group also pauses/terminates uv's Python child instead of only uv itself.
    unsafe {
        match signal {
            ProcessSignal::Pause => {
                libc::kill(-process_id, libc::SIGSTOP);
            }
            ProcessSignal::Resume => {
                libc::kill(-process_id, libc::SIGCONT);
            }
            ProcessSignal::Cancel => {
                // A stopped process does not act on SIGTERM until continued.
                libc::kill(-process_id, libc::SIGCONT);
                libc::kill(-process_id, libc::SIGTERM);
            }
        }
    }
}

#[cfg(target_os = "windows")]
fn signal_process(process_id: u32, signal: ProcessSignal) {
    use std::ffi::c_void;
    use std::process::Command;

    type Handle = *mut c_void;
    const PROCESS_SUSPEND_RESUME: u32 = 0x0800;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn OpenProcess(desired_access: u32, inherit_handle: i32, process_id: u32) -> Handle;
        fn CloseHandle(handle: Handle) -> i32;
    }
    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn NtSuspendProcess(process_handle: Handle) -> i32;
        fn NtResumeProcess(process_handle: Handle) -> i32;
    }

    match signal {
        ProcessSignal::Pause | ProcessSignal::Resume => unsafe {
            let handle = OpenProcess(PROCESS_SUSPEND_RESUME, 0, process_id);
            if !handle.is_null() {
                if matches!(signal, ProcessSignal::Pause) {
                    NtSuspendProcess(handle);
                } else {
                    NtResumeProcess(handle);
                }
                CloseHandle(handle);
            }
        },
        ProcessSignal::Cancel => {
            // taskkill also terminates any helper processes started by uv/git.
            let _ = Command::new("taskkill")
                .args(["/PID", &process_id.to_string(), "/T", "/F"])
                .spawn();
        }
    }
}

#[cfg(not(any(unix, target_os = "windows")))]
fn signal_process(_process_id: u32, _signal: ProcessSignal) {}

static MODEL_DOWNLOAD_CONTROL: DownloadControl = DownloadControl::new();

pub fn model_download_control() -> &'static DownloadControl {
    &MODEL_DOWNLOAD_CONTROL
}

pub fn is_download_cancelled(error: &str) -> bool {
    error.contains(DOWNLOAD_CANCELLED_ERROR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::Arc, thread, time::Duration};

    #[test]
    fn pause_resume_and_cancel_are_observable_at_checkpoints() {
        let control = Arc::new(DownloadControl::new());
        let _session = control.begin();
        assert!(control.pause());

        let worker_control = Arc::clone(&control);
        let worker = thread::spawn(move || worker_control.checkpoint());
        thread::sleep(Duration::from_millis(20));
        assert!(!worker.is_finished());
        assert!(control.resume());
        assert!(worker.join().expect("checkpoint thread").is_ok());

        assert!(control.cancel());
        assert_eq!(
            control.checkpoint().expect_err("cancelled checkpoint"),
            DOWNLOAD_CANCELLED_ERROR
        );
    }
}
