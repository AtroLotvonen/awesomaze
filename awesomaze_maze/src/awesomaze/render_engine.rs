use std::convert::From;
use std::fmt;
use std::io::{self, stdout, Stdout};

use crate::awesomaze::game_model::GameModel;
use crossterm::{
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{prelude::*, widgets::*};

use super::game_model::MazeSquare;

type Tui = Terminal<CrosstermBackend<Stdout>>;

#[derive(Debug)]
pub struct RenderEngine {
    terminal: Tui,
}

impl RenderEngine {
    pub fn init_render_engine() -> Result<RenderEngine, io::Error> {
        enable_raw_mode()?;
        stdout().execute(EnterAlternateScreen)?;
        let terminal = Terminal::new(CrosstermBackend::new(stdout()));
        match terminal {
            Ok(terminal) => Ok(RenderEngine { terminal }),
            Err(e) => Err(e),
        }
    }

    pub fn deinit_render_engine(self) -> io::Result<()> {
        disable_raw_mode()?;
        stdout().execute(LeaveAlternateScreen)?;
        Ok(())
    }

    pub fn render<F>(&mut self, render_fn: F) -> io::Result<CompletedFrame>
    where
        F: FnOnce(&mut Frame),
    {
        self.terminal.draw(|frame| render_fn(frame))
    }
}

struct GameViewModel {
    view_model_str: String,
}

impl GameViewModel {
    const FLOOR: char = ' ';
    const PLAYER_1: char = '웃';
    const CAKE: char = '🎂';
    const RIGHT_PATH: char = ' ';
    const CROSS: char = '╋';
    const LEFT_CROSS: char = '┫';
    const UP_CROSS: char = '┻';
    const RIGHT_CROSS: char = '┣';
    const DOWN_CROSS: char = '┳';
    const LEFT_TOP_CORNER: char = '┏';
    const RIGHT_TOP_CORNER: char = '┓';
    const LEFT_BOTTOM_CORNER: char = '┗';
    const RIGHT_BOTTOM_CORNER: char = '┛';
    const HORIZONTAL_WALL: char = '━';
    const HORIZONTAL_SHORT_WALL_RIGHT: char = '╺';
    const VERTICAL_WALL: char = '┃';
    const VERTICAL_SHORT_WALL_UP: char = '╹';
    const VERTICAL_SHORT_WALL_DOWN: char = '╻';

    fn mazesquare_to_string(game: &GameModel, index: usize) -> String {
        let location = game.index_to_location(index);
        if let Some(tile) = game.maze_square_at_index(index) {
            match tile {
                MazeSquare::Floor => {
                    let right_tile = game.maze_square_at_location((location.0 + 1, location.1));

                    match right_tile {
                        Some(MazeSquare::Floor) => Self::FLOOR.to_string().repeat(2),
                        _ => String::from("  "),
                    }
                }
                MazeSquare::Wall => {
                    let left_tile =
                        game.maze_square_at_location((location.0.wrapping_sub(1), location.1));
                    let upper_tile =
                        game.maze_square_at_location((location.0, location.1.wrapping_sub(1)));
                    let right_tile = game.maze_square_at_location((location.0 + 1, location.1));
                    let below_tile = game.maze_square_at_location((location.0, location.1 + 1));

                    match (left_tile, upper_tile, right_tile, below_tile) {
                        (
                            Some(MazeSquare::Wall),
                            Some(MazeSquare::Wall),
                            Some(MazeSquare::Wall),
                            Some(MazeSquare::Wall),
                        ) => {
                            format!("{}{}", Self::CROSS, Self::HORIZONTAL_WALL)
                        }
                        (
                            Some(MazeSquare::Wall),
                            _,
                            Some(MazeSquare::Wall),
                            Some(MazeSquare::Wall),
                        ) => {
                            format!("{}{}", Self::DOWN_CROSS, Self::HORIZONTAL_WALL)
                        }
                        (
                            _,
                            Some(MazeSquare::Wall),
                            Some(MazeSquare::Wall),
                            Some(MazeSquare::Wall),
                        ) => {
                            format!("{}{}", Self::RIGHT_CROSS, Self::HORIZONTAL_WALL)
                        }
                        (
                            Some(MazeSquare::Wall),
                            Some(MazeSquare::Wall),
                            _,
                            Some(MazeSquare::Wall),
                        ) => {
                            format!("{}{}", Self::LEFT_CROSS, ' ')
                        }
                        (
                            Some(MazeSquare::Wall),
                            Some(MazeSquare::Wall),
                            Some(MazeSquare::Wall),
                            _,
                        ) => {
                            format!("{}{}", Self::UP_CROSS, Self::HORIZONTAL_WALL)
                        }
                        (_, _, Some(MazeSquare::Wall), Some(MazeSquare::Wall)) => {
                            // format!("{}{}", Self::LEFT_TOP_CORNER, Self::HORIZONTAL_WALL)
                            format!("{}{}", Self::LEFT_TOP_CORNER, Self::HORIZONTAL_WALL)
                        }
                        (Some(MazeSquare::Wall), _, _, Some(MazeSquare::Wall)) => {
                            format!("{}{}", Self::RIGHT_TOP_CORNER, ' ')
                        }
                        (Some(MazeSquare::Wall), Some(MazeSquare::Wall), _, _) => {
                            format!("{}{}", Self::RIGHT_BOTTOM_CORNER, ' ')
                        }
                        (_, Some(MazeSquare::Wall), Some(MazeSquare::Wall), _) => {
                            format!("{}{}", Self::LEFT_BOTTOM_CORNER, Self::HORIZONTAL_WALL)
                        }
                        (Some(MazeSquare::Wall), _, Some(MazeSquare::Wall), _) => {
                            format!("{}{}", Self::HORIZONTAL_WALL, Self::HORIZONTAL_WALL)
                        }
                        (_, _, Some(MazeSquare::Wall), _) => {
                            format!(
                                "{}{}",
                                Self::HORIZONTAL_SHORT_WALL_RIGHT,
                                Self::HORIZONTAL_WALL
                            )
                        }
                        (Some(MazeSquare::Wall), _, _, _) => {
                            format!(
                                "{}{}",
                                Self::HORIZONTAL_WALL,
                                ' ' // Self::HORIZONTAL_SHORT_WALL_LEFT
                            )
                        }
                        (_, Some(MazeSquare::Wall), _, Some(MazeSquare::Wall)) => {
                            format!("{}{}", Self::VERTICAL_WALL, ' ')
                        }
                        (_, _, _, Some(MazeSquare::Wall)) => {
                            format!("{}{}", Self::VERTICAL_SHORT_WALL_DOWN, ' ')
                        }
                        (_, Some(MazeSquare::Wall), _, _) => {
                            format!("{}{}", Self::VERTICAL_SHORT_WALL_UP, ' ')
                        }
                        _ => String::from("  "),
                    }
                }
            }
        } else {
            String::from("  ")
        }
    }
}

impl fmt::Display for GameViewModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.view_model_str)
    }
}

impl From<&GameModel> for String {
    fn from(value: &GameModel) -> Self {
        let bounds = value.bounds();
        let player_loc = value.player_location();
        let player_ind = player_loc.1 * bounds.0 + player_loc.0;
        let cake_loc = value.cake_location();
        let cake_ind = cake_loc.1 * bounds.0 + cake_loc.0;
        let mut view_model = String::new();

        for (i, _tile) in value.maze().iter().enumerate() {
            let tile_char = match i {
                i if i == player_ind => GameViewModel::PLAYER_1.to_string(),
                i if i == cake_ind => GameViewModel::CAKE.to_string(),
                i if value.is_in_right_path(i) => GameViewModel::RIGHT_PATH.to_string() + " ",
                _ => GameViewModel::mazesquare_to_string(value, i),
            };
            view_model.push_str(&tile_char);
            if (i + 1) % bounds.0 == 0 {
                view_model.push('\n');
            }
        }
        view_model
    }
}

impl WidgetRef for GameModel {
    fn render_ref(&self, area: Rect, buf: &mut Buffer) {
        let maze: String = self.into();
        let mut text = Vec::new();
        let level_text = format!("Level: {}", self.level());
        for line in maze.lines() {
            let cloned_line = line.green();
            text.push(Line::from(cloned_line))
        }

        let block = Block::default()
            .border_style(Style::default().fg(Color::Blue))
            .borders(Borders::ALL)
            .title(Span::styled(
                level_text,
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .fg(Color::Rgb(255, 192, 203)),
            ))
            .title_alignment(Alignment::Center);
        Paragraph::new(text)
            .centered()
            .block(block)
            .centered()
            .render(area, buf);
    }
}

impl Widget for GameModel {
    fn render(self, area: Rect, buf: &mut Buffer) {
        self.render_ref(area, buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamemodel_to_string_corners() {
        let gm = GameModel::new((20, 30), 1);
        let gm_str: String = (&gm).into();
        for (i, line) in gm_str.lines().enumerate() {
            for (char_i, character) in line.chars().enumerate() {
                // println!("Testing char: {} at ({}, {})", character, i, char_i);
                match (i, char_i) {
                    (0, 0) => assert_eq!(character, GameViewModel::LEFT_TOP_CORNER),
                    (0, 39) => assert_eq!(character, GameViewModel::RIGHT_TOP_CORNER),
                    (29, 0) => assert_eq!(character, GameViewModel::LEFT_BOTTOM_CORNER),
                    (29, 39) => assert_eq!(character, GameViewModel::RIGHT_BOTTOM_CORNER),
                    _ => {}
                }
            }
        }
    }
}
