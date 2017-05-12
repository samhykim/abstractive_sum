import os
import string
import re
import sys
import argparse
from defs import NUM
import random

class Config:
    def __init__(self, args):
        if not args.test:
            self.input_path = args.input_path
            self.output_path = args.output_path
        else:
            self.input_path = "../data_processed/"
            self.output_path = "../data/"
        if self.input_path[-1] != os.sep:
            self.input_path += os.sep
        if self.output_path[-1] != os.sep:
            self.output_path += os.sep
        self.csv = args.csv
        self.test = args.test_size
        self.dev = args.dev_size

def process_data(args):
    config = Config(args)
    print "Input Path: ", config.input_path
    print "Output Path: ", config.output_path
    if not os.path.exists(config.input_path):
        print "Input Path doesn't exist"
        sys.exit(1)
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    if not config.csv:
        with open(config.input_path + "headlines") as f:
            headlines = f.readlines()
        with open(config.input_path + "text") as f:
            text = f.readlines()
        index = [i for i in range(len(headlines))]
        random.shuffle(index)
        dev_h = []
        dev_t = []
        test_h = []
        test_t = []
        train_t = []
        train_h = []
        for i, ind in enumerate(index):
            if i < config.dev:
                dev_h.append(headlines[ind])
                dev_t.append(text[ind])
            elif i < config.test + config.dev:
                test_h.append(headlines[ind])
                test_t.append(text[ind])
            else:
                train_h.append(headlines[ind])
                train_t.append(text[ind])
        print "Training Set Size: ", len(train_h)
        print "Dev Set Size: ", len(dev_h)
        print "Test Set Size: ", len(test_h)
        with open(config.output_path + "dev_headlines", "w") as f_h:
            with open(config.output_path + "dev_text", "w") as f_t:
                for i in range(len(dev_h)):
                    f_h.write(dev_h[i])
                    f_t.write(dev_t[i])
        with open(config.output_path + "test_headlines", "w") as f_h:
            with open(config.output_path + "test_text", "w") as f_t:
                for i in range(len(test_h)):
                    f_h.write(test_h[i])
                    f_t.write(test_t[i])
        with open(config.output_path + "train_headlines", "w") as f_h:
            with open(config.output_path + "train_text", "w") as f_t:
                for i in range(len(train_h)):
                    f_h.write(train_h[i])
                    f_t.write(train_t[i])
    else:
        with open(config.input_path + "data.csv") as f:
            data = f.readlines()
        index = [i for i in range(len(data))]
        random.shuffle(index)
        dev = []
        test = []
        train = []
        for i, ind in enumerate(index):
            if i < config.dev:
                dev.append(data[ind])
            elif i < config.test + config.dev:
                test.append(data[ind])
            else:
                train.append(data[ind])
        print "Training Set Size: ", len(train)
        print "Dev Set Size: ", len(dev)
        print "Test Set Size: ", len(test)

        with open(config.output_path + "dev.csv", "w") as f:
            for i in range(len(dev)):
                f.write(dev[i])
        with open(config.output_path + "test.csv", "w") as f:
            for i in range(len(test)):
                f.write(test[i])
        with open(config.output_path + "train.csv", "w") as f:
            for i in range(len(train)):
                f.write(train[i])

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing Data')
    parser.add_argument('-in', '--input-path', type=str, action='store', default="/media/sanggookang/Gigaword/Processed/", help='Input Path')
    parser.add_argument('-out', '--output-path', type=str, action='store', default='../data_processed/', help='Output Path')
    parser.add_argument('-t', '--test', action="store_true", default=False, help='test functionality with smaller dataset')
    parser.add_argument('-c', '--csv', action="store_true", default=False, help='use the csv file')
    parser.add_argument('-dev', '--dev-size', type=int, action='store', default=120000)
    parser.add_argument('-test', '--test-size', type=int, action='store', default=120000)
    ARGS = parser.parse_args()
    process_data(ARGS)
