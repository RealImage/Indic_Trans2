import re
import sys
import torch
import argparse
from IndicTransToolkit import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
 
# Argument parsing logic
class ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
 
def argumentParser():
    helpStr = "Please provide the input file name, output file name, source language, target language"
    parser = ArgParser(description=helpStr, add_help=True)
    parser.add_argument('--ifile', help="Input file name", required=True)
    parser.add_argument('--ofile', help="Output file name", required=True)
    parser.add_argument('--slang', help="Source Language (Language used in the input file)", required=True)
    parser.add_argument('--tlang', help="Target Language (Language to be used in the output file)", required=True)
    parser.add_argument('--encoding', help="File encoding (default: utf-8)", default='utf-8')
    parsed = parser.parse_args()
    return parsed
 
inputs = argumentParser()
 
# Initialize the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)
ip = IndicProcessor(inference=True)
 
input_file = inputs.ifile
output_file = inputs.ofile
file_encoding = inputs.encoding
 
with open(input_file, "r", encoding=file_encoding) as ifile:
    with open(output_file, 'w', encoding='utf-8') as ofile:
        for line in ifile:
            print(line.strip())  # Print the original line for debugging
            if not re.match(r"\d+\n", line) and not re.match(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", line) and not line.strip() == "":
                # Translate the line
                input_text = line.strip()
               
                # Preprocess the batch for translation
                batch = ip.preprocess_batch([input_text], src_lang=inputs.slang, tgt_lang=inputs.tlang)
                batch = tokenizer(batch, padding="longest", truncation=True, max_length=256, return_tensors="pt")
 
                with torch.inference_mode():
                    outputs = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)
 
                with tokenizer.as_target_tokenizer():
                    # Decode the generated tokens
                    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
 
                # Postprocess the translations
                translated_line = ip.postprocess_batch(outputs, lang=inputs.tlang)[0]
                print(translated_line)  # Print the translated line for debugging
                line = translated_line + "\n"  # Add a newline to the translated line
 
            ofile.write(line)  # Write the line to the output file
