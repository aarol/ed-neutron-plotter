package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

type System struct {
	ID64     int    `json:"id64"`
	Name     string `json:"name"`
	MainStar string `json:"mainStar"`
	Coords   struct {
		X float32 `json:"x"`
		Y float32 `json:"y"`
		Z float32 `json:"z"`
	} `json:"coords"`
	// UpdateTime string `json:"updateTime"`
}

func main() {
	f, err := os.Open("public/systems_neutron.json")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Scan()

	outfiles := make([]*os.File, 0)
	for i := range 4 {
		filename := fmt.Sprintf("public/neutron_coords_%d.bin", i)
		outfile, err := os.Create(filename)
		outfiles = append(outfiles, outfile)
		if err != nil {
			panic(err)
		}
	}

	defer func() {
		for _, outfile := range outfiles {
			outfile.Close()
		}
	}()

	data := []float32{0, 0, 0}
	for scanner.Scan() {
		var sys System
		line := scanner.Bytes()
		if line[len(line)-1] == ']' {
			break
		}
		if line[len(line)-1] == ',' {
			line = line[:len(line)-1]
		}
		if err := json.Unmarshal(line, &sys); err == io.EOF {
			break
		} else if err != nil {
			fmt.Println(string(line))
			panic(err)
		}
		data[0] = sys.Coords.X
		data[1] = sys.Coords.Y
		data[2] = sys.Coords.Z

		i := 0
		if sys.Coords.X > 0 {
			i += 2
		}
		if sys.Coords.Y > 0 {
			i += 1
		}

		err = binary.Write(outfiles[i], binary.LittleEndian, data)
		if err != nil {
			panic(err)
		}
	}

	fmt.Println("Done!")
}
