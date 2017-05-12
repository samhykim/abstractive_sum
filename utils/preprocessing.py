import os
import string
import re
import sys
import argparse
from defs import NUM


class Config:
    def __init__(self, args):
        if not args.test:
            self.input_path = args.input_path
            self.output_path = args.output_path
        else:
            self.input_path = "../data/"
            self.output_path = "../data_processed/"
        if self.input_path[-1] != os.sep:
            self.input_path += os.sep
        if self.output_path[-1] != os.sep:
            self.output_path += os.sep
        self.csv = args.csv
        self.append = args.append

def process_string(line, num_sentences = 3, min_words = 5, max_words = 200):
    #text = re.sub("([0-9])+.+([0-9])", NUM, text)
    article = ""
    sentence_counter = 0
    line = line.replace("\'\'", "\"")
    line = line.replace("``", "\"")
    line = line.replace("u.s.a", "usa")
    line = line.replace("u.s.", "usa")
    line = line.replace("u.n.", "un")
    for i, char in enumerate(line):
        if char == "-" and i > 0 and line[i - 1].isdigit() and i < len(line) - 1 and line[i + 1].isdigit():
            continue
        if char.isdigit() and i < len(line) - 1 and line[i + 1].isalpha():
            article += " "
        if char in ",.;\"\'][}{/)(":
            if char in '.,' and i < len(line) - 1 and line[i + 1].isdigit() and i > 0 and line[i - 1].isdigit():
                continue
            elif char == ".":
                sentence_counter += 1
            if i >= len(line) - 2 or line[i + 1] in ",.;\"\'][}{/)(" or i < 2 or (line[i - 1] != " " and line[i - 2] != '.' and (line[i + 1] == " " or (line[i + 1] != " " and line[i + 2] not in  ' .'))):
                article += " "
            article += char
            if i < 2 or ((line[i - 1] == " " or (line[i - 1] != " " and line[i - 2] != '.')) and ((i == len(line) - 2 and line[i + 1] != " ") or (i < len(line) - 2 and (line[i + 1].isdigit() or (line[i + 1].isalpha() and line[i + 2] not in ". "))))):
                article += " "
        else: article += char
        if sentence_counter == num_sentences:
            break
    article = string.replace(article, "\'", " \'")
    split_article = article.split()
    if len(split_article) < min_words or len(split_article) > max_words:
        return None
    return " ".join(split_article)

def process_file(path, headline, text):
    with open(path, 'rt') as f:
        print "opened file", path
        line = f.readline().rstrip('\n')
        while line:
            while line != "<HEADLINE>" and line:
                line = f.readline().rstrip('\n')
            if not line:
                break
            line = f.readline().rstrip('\n')
            if not line:
                break
            title = line.lower()
            article = ""
            while line != "<TEXT>" and line:
                line = f.readline().rstrip('\n')
            if not line:
                break
            line = f.readline().rstrip('\n')
            
            while line != "</TEXT>" and line:
                if line != "<P>" and line != "</P>":
                    line = line.lower()
                    article += line + " "
                line = f.readline().rstrip('\n')
            if len(article) == 0:
                continue
            title = process_string(title.lower(), num_sentences = float("inf"))
            article = process_string(article.lower())
            if title == None or article == None:
                continue
            if text != None:
                headline.write(title + '\n')
                text.write(article + '\n')
            else:
                headline.write(title + '\t' + article + '\n')
          

def process_data(args):
    config = Config(args)
    print "Input Path: ", config.input_path
    print "Output Path: ", config.output_path
    if not os.path.exists(config.input_path):
        print "Input Path doesn't exist"
        sys.exit(1)
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    for subdir, dirs, files in os.walk(config.input_path):
        if "data" in subdir:
            relDir = os.path.relpath(subdir, config.input_path)
            out_subdir = config.output_path + relDir + os.sep
            if not os.path.exists(out_subdir):
                os.makedirs(out_subdir)
            if not config.csv:
                if not config.append:
                    with open(config.output_path + relDir + os.sep + 'headlines', 'w') as headline:
                        with open(config.output_path + relDir + os.sep + 'text', 'w') as text: 
                            for file in files:
                                process_file(subdir + os.sep + file, headline, text)
                else:
                    with open(config.output_path + 'headlines', 'a') as headline:
                        with open(config.output_path + 'text', 'a') as text: 
                            for file in files:
                                process_file(subdir + os.sep + file, headline, text)
            else:
                if not config.append:
                    with open(config.output_path + relDir + os.sep + 'data.csv', 'w') as data:
                        for file in files:
                            process_file(subdir + os.sep + file, data, None)
                else:
                    with open(config.output_path + 'data.csv', 'a') as data:
                        for file in files:
                            process_file(subdir + os.sep + file, data, None)

#if not os.path.exists(parsed_path):
#    os.makedirs(parsed_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing Data')
    parser.add_argument('-in', '--input-path', type=str, action='store', default="/media/sanggookang/Gigaword/LDC2011T07_English-Gigaword-Fifth-Edition", help='Input Path')
    parser.add_argument('-out', '--output-path', type=str, action='store', default='/media/sanggookang/Gigaword/Processed', help='Output Path')
    parser.add_argument('-t', '--test', action="store_true", default=False, help='test functionality with smaller dataset')
    parser.add_argument('-c', '--csv', action="store_true", default=False, help='store as a csv file')
    parser.add_argument('-a', '--append', action="store_true", default=False, help='store in a single directory')
    ARGS = parser.parse_args()
    process_data(ARGS)
