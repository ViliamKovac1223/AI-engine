package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
)

var (
	defaultLibOut     = "./example/include/tensor.hpp"
	defaultDefineImpl = "TENSOR_LIB_IMPL"
)

func main() {
	root := flag.String("root", ".", "project root")
	out := flag.String("out", defaultLibOut, "output single header")
	impl := flag.String("impl-macro", defaultDefineImpl, "implementation macro name")

	if root == nil || out == nil || impl == nil {
		fmt.Println("Error in processing arguments")
		return
	}
	flag.Parse()

	_, err := makeSingleHeader(*root, *out, *impl)
	if (err != nil) {
		fmt.Printf("error: %s\n", err)
	}
}

func makeSingleHeader(root string, des string, defineImplPrefix string) (string, error) {
	// Get all files from src and include folders
	folders := []string{root + "/src", root + "/include"}
	files, err := getAllFilesInfo(&folders)
	if err != nil {
		return "", nil
	}

	// Get all lines of code from headers and source files
	// Lines were processed, when creating FileInfo structure
	// So they're in correct format
	headersText := ""
	sourcesText := ""
	for _, file := range *files {
		if file.Type == Source {
			sourcesText += file.Content
		} else if file.Type == Header {
			headersText += file.Content
		} else if file.Type == Unsopported {
			fmt.Printf("Unsupported file (%s)\n", file.Path)
		}
	}

	desFile, err := os.OpenFile(des, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		fmt.Println("Cannot open destination file: " + des)
		return "", err
	}

	// Define guard strings
	ifdef := "\n#ifndef " + defineImplPrefix + "\n"
	def := "#define " + defineImplPrefix + "\n"
	enddef := "\n#endif\n"

	// Write headers' content and source files' content into destination file
	for _, fileContent := range []*string{
		&headersText, // Headers
		// Wrap sources between implementation prefix guard
		&ifdef, &def,
		&sourcesText,
		&enddef,
	} {
		_, err = desFile.Write([]byte(*fileContent))
		if err != nil {
			fmt.Println("Cannot write to destination file: " + des)
			return "", err
		}
	}

	defer desFile.Close()

	return "", nil
}

func getAllFilesInfo(folders *[]string) (*[]FileInfo, error) {
	fileTypes := []FileInfo{}

	files, err := findAllFiles(folders)
	if err != nil {
		return nil, err
	} else if files == nil {
		return nil, errors.New("Files not found")
	}

	for _, file := range *files {
		fileType, err := NewFileType(file)
		if err != nil {
			return nil, err
		}

		fileTypes = append(fileTypes, *fileType)
	}

	return &fileTypes, nil
}

func findAllFiles(folders *[]string) (*[]string, error) {
	files := []string{}
	if folders == nil {
		return &files, errors.New("No folders to search")
	}

	var err error
	for _, folder := range *folders {
		err = filepath.WalkDir(folder, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				// Skip unreadable files
				return nil
			}

			if !d.IsDir() {
				files = append(files, path)
			}
			return nil
		})
	}

	return &files, err
}
