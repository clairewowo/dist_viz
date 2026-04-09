use hdf5::File;
use rand::prelude::*;
use std::env;
use anndists::dist::DistL2;
use rand::thread_rng;
mod utils;
use utils::*;
use rayon::prelude::*;
use plotters::prelude::*;
use std::sync::Mutex;

fn euclid(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for j in 0..a.len() {
        let d = a[j] - b[j];
        s += d * d;
    }
    s.sqrt()
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let filename = args[1].clone();
    let anndata = annhdf5::AnnBenchmarkData::new(filename.clone()).unwrap_or_else(|_| {
        panic!(
            "Failed to load {:?}. Delete and regenerate the file if needed.",
            filename
        )
    });
    let sample_size = 5000;

    // sample points
    let mut rng = thread_rng();
    let sample: Vec<Vec<f32>> = anndata.train_data
        .choose_multiple(&mut rng, sample_size) // pick references to tuples
        .map(|(vec_f32, _)| vec_f32.clone()) // extract Vec<f32> and clone
        .collect();

    // Define number of bins
    let bins = 20;
    let bin_width = 1.0 / bins as f32;

    // calculate pairwise distances, place into bins (0 to 19)
    let mut counts = Mutex::new(vec![0; bins]);
    sample
        .par_iter()
        .for_each(|v1| {
            for v2 in &sample {
                if v1 == v2 {
                    continue;
                }
                let dist = euclid(v1, v2);
                let bin = (dist / bin_width).floor().min(bins as f32 - 1.0) as usize;
                let mut counts_lock = counts.lock().unwrap();
                counts_lock[bin] += 1;
            }
        });
    let counts = counts.into_inner().unwrap();

    let output = filename + "_dist_plot.png";

    // plot histogram
    let root = BitMapBackend::new(&output, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_count = *counts.iter().max().unwrap_or(&1);
    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Distance distribution of {}", output), ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..bins as i32, 0..max_count as i32)?;
    
    chart.configure_mesh().draw()?;
    // Draw bars
    // for (i, &count) in counts.iter().enumerate() {
    //     let x0 = i as f32 * bin_width;
    //     let x1 = x0 + bin_width;
    //     
    //     chart.draw_series(std::iter::once(Rectangle::new(
    //         [(x0 as i32, 0), (x1 as i32, count as i32)],
    //         BLUE.filled(),
    //     )))?;
    // }

    for (i, &count) in counts.iter().enumerate() {
        println!("count for bin {}: {}", i, count);
        chart.draw_series(std::iter::once(Rectangle::new(
            [(i as i32, 0), (i as i32 + 1, count as i32)],
            BLUE.filled(),
        )))?;
    }

    
    root.present()?;
    println!("Histogram saved to {}", output);
    Ok(())
}
