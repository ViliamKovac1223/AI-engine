package main

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"regexp"
	"strings"
)

type FileType int

const (
	Source FileType = iota
	Header
	Unsopported
)

type FileInfo struct {
	Path    string
	Type    FileType
	Content string
}

var (
	includeRe = regexp.MustCompile(`#include\s+".*".*$`)
)

func NewFileType(path string) (*FileInfo, error) {
	fileType := getFileType(path)
	content, err := getFileFilteredCotent(path, fileType)
	if err != nil || content == nil {
		return nil, err
	}

	return &FileInfo{
		Path:    path,
		Type:    fileType,
		Content: *content,
	}, nil
}

func getFileFilteredCotent(path string, fileType FileType) (*string, error) {
	// Read file content
	content, err := os.ReadFile(path)
	if err != nil {
		fmt.Println("Error reading file:", err)
		return nil, errors.New("Couldn't read file: " + path)
	}
	contentStr := string(content)

	// Filter out includes from source files
	if fileType == Source {
		return removeIncludes(&contentStr), nil
	}

	return &contentStr, nil
}

func removeIncludes(content *string) *string {
	if content == nil {
		panic("File content is nil")
	}

	var builder strings.Builder
	scanner := bufio.NewScanner(strings.NewReader(*content))

	for scanner.Scan() {
		line := scanner.Text()
		if !includeRe.MatchString(line) {
			builder.WriteString(line)
			builder.WriteByte('\n') // Add new line to the string
		}
	}

	result := strings.TrimSuffix(builder.String(), "\n")

	return &result
}

func getFileType(path string) FileType {
	fileExtensions := map[string]FileType{
		"cpp": Source,
		"hpp": Header,
	}

	for extension, fileType := range fileExtensions {
		pattern := fmt.Sprintf(`\.%s$`, extension)
		matched, _ := regexp.MatchString(pattern, path)
		if matched {
			return fileType
		}
	}

	return Unsopported
}
