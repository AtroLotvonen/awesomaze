// engine.rs

use std::io;

use ratatui::Frame;

use super::input::GameInput;

pub trait Engine {
    fn tick(&mut self, input: Option<GameInput>) -> io::Result<bool>;
    fn render_frame(&self, frame: &mut Frame);
}
