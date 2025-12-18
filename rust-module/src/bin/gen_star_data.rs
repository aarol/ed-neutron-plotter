use std::io::{self, Read, Write};

use rkyv::rancor::Error;
use rust_module::{
    fast_json_parser::SystemParser,
    star::Star,
    trie::{CompactRadixTrie, TrieBuilder},
};


#[allow(dead_code)]
fn analyze() -> io::Result<()> {
    let mut file = std::io::BufReader::new(std::fs::File::open("../public/data/search_trie.bin")?);

    let mut buf = vec![];
    file.read_to_end(&mut buf)?;
    let trie = CompactRadixTrie::from_bytes(&buf);
    println!("Trie has {} nodes", trie.nodes.len());

    // Test contains
    println!("\nContains 'Colonia': {}", trie.contains("Colonia"));

    // Test suggest with different prefixes
    println!("\nSuggestions for 'Col':");
    for suggestion in trie.suggest("Col", 10) {
        println!("  - {}", suggestion);
    }

    println!("\nSuggestions for 'Colonia':");
    for suggestion in trie.suggest("Colonia", 10) {
        println!("  - {}", suggestion);
    }

    println!("\nSuggestions for 'Speam':");
    for suggestion in trie.suggest("Speam", 10) {
        println!("  - {}", suggestion);
    }

    Ok(())
}

fn main() -> io::Result<()> {
    // Uncomment to analyze existing trie:
    // return analyze();

    let out_dir = std::path::Path::new("../public/data");
    std::fs::create_dir_all(out_dir)?;

    let neutrons = std::fs::File::open("systems_neutron.json")?;

    let systems = std::fs::File::open("systems_1day.json")?;

    let mut stars = vec![];
    let mut trie = TrieBuilder::new();

    let parser = SystemParser::new(neutrons)?;

    let parser2 = SystemParser::new(systems)?;

    let mut count = 0;
    parser.for_each(|name, coords| {
        if count % 100000 == 0 {
            println!("Processing line {}", count);
        }
        count += 1;

        // The coordinates are in light years, three.js doesn't like such huge distances
        // This will reduce the scale to max [-100, 100] in each axis
        stars.push(Star::new(
            -(coords.x / 1000.0) as f32,
            (coords.y / 1000.0) as f32,
            (coords.z / 1000.0) as f32,
        ));

        trie.insert(name);
    })?;

    parser2.for_each(|name, _coords| {
        if count % 100000 == 0 {
            println!("Processing line {}", count);
        }
        count += 1;

        // Don't show these yet
        // stars.push(Star::new(
        //     (coords.x / 1000.0) as f32,
        //     (coords.y / 1000.0) as f32,
        //     (coords.z / 1000.0) as f32,
        // ));

        trie.insert(name);
    })?;

    let star_coords: Vec<[f32; 3]> = stars.iter().map(|s| [s.x, s.y, s.z]).collect();
    let kdtree = kiddo::ImmutableKdTree::new_from_slice(&star_coords);
    
    let kdtree_bytes = rkyv::to_bytes::<Error>(&kdtree).unwrap();
    let mut kdtree_file = std::fs::File::create(out_dir.join("star_kdtree.bin"))?;
    kdtree_file.write_all(&kdtree_bytes)?;
    // partitions.par_iter().enumerate().for_each(|(i, p)| {
    //     let mut file =
    //         std::fs::File::create(out_dir.join(format!("neutron_stars{}.bin", i))).unwrap();

    //     p.write_to_file(&mut file).unwrap();
    // });

    let (nodes, labels) = trie.build();
    let trie = CompactRadixTrie::new(&nodes, &labels);

    dbg!(trie.suggest("Speam", 10));

    trie.analyze_stats();
    println!("Uses {} MB of space", trie.size_in_bytes() / 1024 / 1024);

    // Write trie to file
    let mut trie_file = std::fs::File::create(out_dir.join("search_trie.bin"))?;

    trie_file.write_all(&trie.to_bytes())?;

    Ok(())
}
