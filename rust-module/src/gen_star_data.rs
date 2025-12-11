mod trie;

use std::io::{self, BufRead, BufReader, Seek, SeekFrom, Write};

use byteorder::WriteBytesExt;
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

use trie::CompactPatriciaTrie;

fn main() -> io::Result<()> {
    let mut file = std::fs::File::open("src/systems_neutron.json")?;
    file.seek(SeekFrom::Start(1))?;
    let reader = BufReader::new(file);

    let star_vecs = &mut [vec![], vec![], vec![], vec![]];

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

            let mut index = 0;
            if system.coords.x >= 0.0 {
                index += 2;
            }
            if system.coords.z >= 25000.0 {
                index += 1;
            }

            star_vecs[index].push(system.coords.x);
            star_vecs[index].push(system.coords.y);
            star_vecs[index].push(system.coords.z);

            trie.insert(system.name.as_str());
        });

    [0, 1, 2, 3].iter().for_each(|i| {
        let mut file = std::fs::File::create(format!("out/neutron_stars{}.bin", i)).unwrap();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                star_vecs[*i].as_ptr() as *const u8,
                star_vecs[*i].len() * std::mem::size_of::<f32>(),
            )
        };
        file.write_all(bytes).unwrap();
    });

    dbg!(trie.contains("Speamoo AA-A h0"));

    println!("Trie has {} nodes", trie.nodes.len());
    println!(
        "Uses {} MB of space",
        trie.nodes.len() * std::mem::size_of::<trie::Node>() / 1024 / 1024
            + trie.labels.len() / 1024 / 1024
    );

    // Write trie to file
    let mut trie_file = std::fs::File::create("out/trie.bin")?;

    println!("1");
    trie_file.write_u32::<byteorder::LittleEndian>(trie.nodes.len() as u32)?;
    let trie_bytes = unsafe {
        std::slice::from_raw_parts(
            trie.nodes.as_ptr() as *const u8,
            trie.nodes.len() * std::mem::size_of::<trie::Node>(),
        )
    };
    println!("2");
    trie_file.write_all(trie_bytes)?;

    println!("3");
    trie_file.write_u32::<byteorder::LittleEndian>(trie.labels.len() as u32)?;
    let trie_label_bytes = unsafe {
        std::slice::from_raw_parts(
            trie.labels.as_ptr() as *const u8,
            trie.labels.len() * std::mem::size_of::<u8>(),
        )
    };
    println!("4");
    trie_file.write_all(trie_label_bytes)?;

    Ok(())
}
