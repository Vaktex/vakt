use std::process::Command;

const LIMIT: usize = 10;

/// Runs a command.
#[inline]
fn run(cmd: &str) -> std::io::Result<()> {
    Command::new("sh").arg("-c").arg(cmd).status().map(|_| ())
}

struct Buf { data: Vec<u8> }

impl Buf {
    fn get(&self, i: usize) -> u8 {
        unsafe { *self.data.get_unchecked(i) }
    }
}
