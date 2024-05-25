import re

input_file = "C:\\Users\\MFE\\Xavier\\requirements_raw.txt"
output_file = "C:\\Users\\MFE\\Xavier\\requirements.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        if re.match(r"^[a-zA-Z0-9\-_]+==", line):
            outfile.write(line)
