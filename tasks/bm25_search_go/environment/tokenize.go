package main

import "strings"

func Tokenize(text string) []string {
	text = strings.ToLower(text)
	return strings.Fields(text)
}
