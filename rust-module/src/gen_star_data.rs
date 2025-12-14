mod star;
mod trie;

use std::io::{self, BufRead, BufReader, Seek, SeekFrom, Write};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize, Debug)]
pub struct System {
    // id: i64,
    // id64: i64,
    name: String,
    coords: Coords,
    // date: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Coords {
    x: f32,
    y: f32,
    z: f32,
}

impl Coords {
    pub fn write_to_file<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&self.x.to_le_bytes())?;
        writer.write_all(&self.y.to_le_bytes())?;
        writer.write_all(&self.z.to_le_bytes())?;
        Ok(())
    }
}

use trie::CompactPatriciaTrie;

use crate::star::{reorder_for_partitions, Partition, Star};

fn main() -> io::Result<()> {
    let mut file = std::fs::File::open("systems_neutron.json")?;
    file.seek(SeekFrom::Start(1))?;
    let reader = BufReader::new(file);

    let mut stars = vec![];

    let mut trie = CompactPatriciaTrie::new();

    reader
        .lines()
        .skip(1)
        // .take(100)
        .enumerate()
        .for_each(|(i, line)| {
            if i % 100000 == 0 {
                println!("Processing line {}", i);
            }

            let mut line = line.unwrap();
            if line == "]" {
                return;
            }
            if line.ends_with(',') {
                line.pop();
            }
            let line = line.as_mut_str();
            let system = unsafe { simd_json::from_str::<System>(line).unwrap() };

            // The coordinates are in light years, three.js doesn't like such huge distances
            // This will reduce the scale to max [-100, 100] in each axis
            stars.push(Star::new(
                system.coords.x / 1000.0,
                system.coords.y / 1000.0,
                system.coords.z / 1000.0,
            ));

            trie.insert(system.name.as_str());
        });
    trie.compress();

    let out_dir = std::path::Path::new("../public/data");
    std::fs::create_dir_all(out_dir)?;

    let k_splits = 3;
    let num_partitions = 1 << k_splits;

    reorder_for_partitions(&mut stars, k_splits);

    let chunk_size = stars.len() / num_partitions;
    let partitions: Vec<Partition> = stars
        .chunks(chunk_size)
        .map(|p| Partition::new(p))
        .collect();

    println!(
        "Created {} partitions of size approx {}",
        partitions.len(),
        chunk_size
    );

    partitions.par_iter().enumerate().for_each(|(i, p)| {
        let mut file =
            std::fs::File::create(out_dir.join(format!("neutron_stars{}.bin", i))).unwrap();

        p.write_to_file(&mut file).unwrap();
    });

    dbg!(trie.suggest("Speamoo", 10));

    println!("Trie has {} nodes", trie.nodes.len());
    println!("Uses {} MB of space", trie.size_in_bytes() / 1024 / 1024);

    let label_bytes_before = trie.labels.len();

    // trie.deduplicate_labels();

    println!(
        "After deduplication, uses {} MB of space (labels reduced from {} to {} bytes)",
        trie.nodes.len() * std::mem::size_of::<trie::Node>() / 1024 / 1024
            + trie.labels.len() / 1024 / 1024,
        label_bytes_before,
        trie.labels.len()
    );

    // Write trie to file
    let mut trie_file = std::fs::File::create(out_dir.join("search_trie.bin"))?;

    trie_file.write_all(&trie.to_bytes())?;

    Ok(())
}
