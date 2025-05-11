// info.rs

use std::{collections::VecDeque, time::Instant};

use ratatui::{prelude::*, text::Span, widgets::*};

use ratatui::{
    style::Color,
    widgets::{Block, Borders, Chart, Dataset, Widget},
};

use ratatui::{buffer::Buffer, layout::Rect, style::Style, symbols};

#[derive(Clone, Debug)]
pub struct Info {
    current_iter: usize,
    pub learning_rate: f32,
    pub exploration_rate: f32,
    max_iters: usize,
    finished_history: VecDeque<f32>,
    finished_average: Vec<(f64, f64)>,
    reward_history_len: usize,
    loss_history: Vec<(f64, f64)>,
    start_time: Instant,
    eta: Option<(u64, u64)>,
}

impl Info {
    pub fn new(max_iters: usize) -> Info {
        Info {
            current_iter: 0,
            learning_rate: 0.0,
            exploration_rate: 0.0,
            max_iters,
            finished_history: VecDeque::new(),
            finished_average: Vec::new(),
            reward_history_len: 0,
            loss_history: Vec::new(),
            start_time: Instant::now(),
            eta: None,
        }
    }

    pub fn add_loss(&mut self, amount: usize, loss: f32) {
        self.current_iter += amount;
        self.loss_history
            .push((self.current_iter as f64, loss as f64));
        let elapsed_time = self.start_time.elapsed().as_secs_f64();
        let estimated = (elapsed_time / self.current_iter as f64) * self.max_iters as f64;
        let eta = (estimated - elapsed_time) as u64;
        let eta_hours = eta / 3600;
        let eta_mins = (eta % 3600) / 60;
        self.eta = Some((eta_hours, eta_mins));
    }

    pub fn add_rewards(&mut self, rewards: Vec<f32>) {
        self.finished_history.extend(rewards);
        while self.finished_history.len() >= 100 {
            let chunk = self.finished_history.drain(0..100);
            self.finished_average.push((
                (self.reward_history_len * 100) as f64,
                (chunk.sum::<f32>() / 100.0).into(),
            ));
            self.reward_history_len += 1;
        }
    }
}

impl Widget for Info {
    fn render(self, area: ratatui::prelude::Rect, buf: &mut ratatui::prelude::Buffer)
    where
        Self: Sized,
    {
        self.render_ref(area, buf)
    }
}

impl WidgetRef for Info {
    fn render_ref(&self, info_area: Rect, buf: &mut Buffer) {
        let [up, down] = Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)])
            .areas(info_area);
        // Draw the block with borders and title
        let block = Block::default()
            .title("Average reward (100)")
            .title_alignment(Alignment::Left)
            .borders(Borders::ALL);
        block.render(up, buf);
        // Get the data and calculate the maximum and minimum for scaling
        if !self.finished_average.is_empty() {
            let datasets = vec![Dataset::default()
                .name(format!(
                    "Current: {:.5}",
                    &self.finished_average.iter().last().unwrap().1
                ))
                .marker(symbols::Marker::Dot)
                .style(Style::default().fg(Color::LightBlue))
                .data(&self.finished_average)];
            let chart = Chart::new(datasets)
                .block(
                    Block::bordered().title(Span::styled(
                        "Rewards",
                        Style::default()
                            .fg(Color::LightBlue)
                            .add_modifier(Modifier::BOLD),
                    )),
                )
                .x_axis(
                    Axis::default()
                        .title("Step")
                        .style(Style::default().fg(Color::Gray))
                        .bounds([0.0, (self.reward_history_len * 100) as f64])
                        .labels(vec![
                            Span::styled("0", Style::default().add_modifier(Modifier::BOLD)),
                            Span::styled(
                                format!("{}", self.reward_history_len * 100),
                                Style::default().add_modifier(Modifier::BOLD),
                            ),
                        ]),
                )
                .y_axis(
                    Axis::default()
                        .title("Reward")
                        .style(Style::default().fg(Color::Gray))
                        .bounds([-4.0, 4.0])
                        .labels(vec![
                            Span::styled(
                                format!("{:.1}", -4.0),
                                Style::default().add_modifier(Modifier::BOLD),
                            ),
                            Span::raw("0.0"),
                            Span::styled(
                                format!("{:.1}", 4.0),
                                Style::default().add_modifier(Modifier::BOLD),
                            ),
                        ]),
                );

            chart.render(up, buf);
        }

        let [left, right] =
            Layout::horizontal([Constraint::Percentage(30), Constraint::Percentage(70)])
                .areas(down);
        let current_iter = self.current_iter;
        let lr = self.learning_rate;
        let exploration_rate = self.exploration_rate;
        let iter_line =
            Line::from(format!("Current iter: {current_iter}")).alignment(Alignment::Left);
        let max_iter_line =
            Line::from(format!("Max iter: {}", self.max_iters)).alignment(Alignment::Left);
        let lr_line = Line::from(format!("Learning rate: {lr}")).alignment(Alignment::Left);
        let exploration_rate_line =
            Line::from(format!("Exploration rate: {exploration_rate}")).alignment(Alignment::Left);
        let duration = self.start_time.elapsed();
        let elapsed_hours = duration.as_secs() / 3600;
        let elapsed_minutes = (duration.as_secs() % 3600) / 60;
        let elapsed_time_line = Line::from(format!(
            "Elapsed time: {:?} h {:?} mins",
            elapsed_hours, elapsed_minutes
        ))
        .alignment(Alignment::Left);
        let eta_h_min = if let Some(eta) = self.eta {
            format!("Eta: {:?} h {:?} mins", eta.0, eta.1)
        } else {
            "-".to_string()
        };
        let eta_line = Line::from(eta_h_min).alignment(Alignment::Left);
        let text = Text::from(vec![
            iter_line,
            max_iter_line,
            lr_line,
            exploration_rate_line,
            elapsed_time_line,
            eta_line,
        ]);
        let block = Paragraph::new(text).block(Block::bordered().title_top("Info"));
        block.render(left, buf);

        let block = Block::default()
            .title("Loss History")
            .title_alignment(Alignment::Left)
            .borders(Borders::ALL);
        block.render(right, buf);

        if !self.loss_history.is_empty() {
            let datasets = vec![Dataset::default()
                .name(format!(
                    "Current: {:.5}",
                    &self.loss_history.last().unwrap().1
                ))
                .marker(symbols::Marker::Dot)
                .style(Style::default().fg(Color::LightRed))
                .data(&self.loss_history)];
            let chart = Chart::new(datasets)
                .block(
                    Block::bordered().title(Span::styled(
                        "Loss History",
                        Style::default()
                            .fg(Color::LightRed)
                            .add_modifier(Modifier::BOLD),
                    )),
                )
                .x_axis(
                    Axis::default()
                        .title("Step")
                        .style(Style::default().fg(Color::Gray))
                        .bounds([0.0, self.current_iter as f64])
                        .labels(vec![
                            Span::styled("0", Style::default().add_modifier(Modifier::BOLD)),
                            Span::styled(
                                format!("{}", self.current_iter),
                                Style::default().add_modifier(Modifier::BOLD),
                            ),
                        ]),
                )
                .y_axis(
                    Axis::default()
                        .title("Loss")
                        .style(Style::default().fg(Color::Gray))
                        .bounds([0.0, 0.2])
                        .labels(vec![
                            Span::styled(
                                format!("{:.1}", 0.0),
                                Style::default().add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(
                                format!("{:.1}", 0.2),
                                Style::default().add_modifier(Modifier::BOLD),
                            ),
                        ]),
                );

            chart.render(right, buf);
        }
        // Get the data and calculate the maximum and minimum for scaling
        // if !self.loss_history.is_empty() {
        //     let data = &self.loss_history;
        //     let area = right;
        //     let latest_value = data.iter().last().copied().unwrap_or(0.0);
        //     let max_value = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        //     let min_value = data.iter().cloned().fold(f32::INFINITY, f32::min);
        //
        //     // Calculate scaling factors
        //     let width = area.width as usize;
        //     let height = area.height as f32;
        //     let range = max_value - min_value;
        //
        //     for (i, &value) in data.iter().rev().enumerate() {
        //         if i >= width - 2 {
        //             break; // Stop if we exceed area width
        //         }
        //         let x = area.right() - i as u16 - 2;
        //         let y = area.bottom()
        //             - ((value - min_value) / range * (height - 2.0)).round() as u16
        //             - 1;
        //
        //         // Draw each point as a dot
        //         buf.get_mut(x, y)
        //             .set_symbol(symbols::DOT)
        //             .set_fg(Color::LightRed);
        //     }
        //
        //
        //     // Display the latest value as a label at the top right of the chart
        //     let label = format!("Latest: {:.5}", latest_value);
        //     let label_x = area.right().saturating_sub(label.len() as u16 + 1);
        //     for (i, ch) in label.chars().enumerate() {
        //         buf.get_mut(label_x + i as u16, area.top())
        //             .set_char(ch)
        //             .set_fg(Color::White);
        //     }
        // }
    }
}
