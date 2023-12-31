import os
import shlex
import pandas as pd
from enum import Enum


class Operation(Enum):
    inRange = 0
    le = 1
    leq = 2
    ge = 3
    geq = 4
    eq =54

from astropy.table import Table

def read_fits(fits_path, hdu=1):
    dat = Table.read(fits_path, format='fits', hdu=hdu)
    names = [name for name in dat.colnames if len(dat[name].shape) <= 1]
    df = dat[names].to_pandas()
    if 'MANGAID' in df:
        df['MANGAID'] = df['MANGAID'].str.decode('utf-8')
        df['MANGAID'] = df['MANGAID'].str.strip()
    return df


def clean_data(config):
    #print(config)

    filePath = "./data/" + config["fileName"] + "." + config["type"]

    df = None

    #Read input file
    if config["type"] == "fits":
       df_list = []
       for hdu in config["hdu"]:
           df_list += [read_fits(filePath, hdu)]

       df = pd.concat(df_list, axis=1)

    for rule in config["filter"]:
        dropping = False
        contains = False
        base = 0

        #Remove instead of select for
        if rule[base] == "DROP":
            dropping = True
            base += 1
            if rule[base] == "NA":
                df = df.dropna(axis=1)
                continue

        #Partial matching 
        if rule[base] == "CONTAINS":
            contains = True
            base += 1

        term = rule[base]
        base += 1

        op = None
        if rule[base] == "IN" and rule[base+1] == "RANGE":
            op = Operation.inRange
            base += 2
        else:
            if rule[base] == "<":
                op = Operation.le
            elif rule[base] == "<=":
                op = Operation.leq
            elif rule[base] == ">":
                op = Operation.ge
            elif rule[base] == ">=":
                op = Operation.geq
            elif rule[base] == "==":
                op = Operation.eq
            else:
                print("Unsupported operation! ", rule)
            base += 1

        applyAll = False 
        if "\"" in term:
            term = term.strip("\"")
        elif term == "ALL":
            applyAll = True

        if applyAll:
            filtered_columns = df.columns 
        elif contains:
            filtered_columns = [col for col in df.columns if term in col]
        else:
            filtered_columns = [col for col in df.columns if term == col]

        if op == Operation.inRange:
            range = rule[base].split(',')

            mask = ((df[filtered_columns] >= float(range[0])) 
                    & (df[filtered_columns] <= float(range[1]))).all(axis=1)

            if dropping:
                df = df[~mask]
            else:
                df = df[mask]
        
        if op == Operation.eq:
            mask = (df[filtered_columns] == float(rule[base])).all(axis=1)
            if dropping:
                df = df[~mask]
            else:
                df = df[mask]

        elif op == Operation.ge:
            if dropping:
                if applyAll:
                    mask = df.map(lambda x: isinstance(x, (int, float)) and x > float(rule[base])).any(axis=1)
                else:
                    mask = (df[filtered_columns] > float(rule[base])).all(axis=1)
            else:
                if applyAll:
                    mask = df.map(lambda x: isinstance(x, (int, float)) and x <= float(rule[base])).any(axis=1)
                else:
                    mask = (df[filtered_columns] <= float(rule[base])).all(axis=1)

            df = df[~mask]

    return df

def select_columns(df, config):
    columns = []

    for rule in config["select"]:
        base = 0
        contains = False

        if rule[base] == "CONTAINS":
            contains = True
            base += 1

        term = rule[base]
        base += 1

        applyAll = False 
        if "\"" in term:
            term = term.strip("\"")
        elif term == "ALL":
            applyAll = True

        if contains:
            columns += ([col.lower() for col in df.columns if term in col ])#Galaxy Zoo]
        else:
            columns += ([col.lower() for col in df.columns if term == col ])#Galaxy Zoo]

    return list(set(columns))

def parse_config(filePath):

    config = {}
    config["fileName"] = (os.path.splitext(filePath.split('/')[-1])[0])

    filterRules = []
    selectRules = []
    metaRules = []

    filtering = False
    selecting = False


    with open(filePath) as f:
        for line in f.readlines():
            # Remove comments by ignoring text after '#'
            line = line.split('#', 1)[0].strip()
            if line:  # Check if the line is not empty after removing comments
                tokens = shlex.split(line, posix=False)

                if tokens[0] == "TYPE":
                    config['type'] = tokens[1]
                elif tokens[0] == "HDU":
                    config['hdu'] = ([int(x) for x in tokens[1].split(',')])
                elif tokens[0] == "LABEL":
                    config['label'] = tokens[1]
                elif tokens[0] == "META":
                    metaRules.append(tokens[1:])
                elif tokens[0] == "FILTER":
                    filterRules.append(tokens[1:])
                elif tokens[0] == "SELECT":
                    selectRules.append(tokens[1:])

    config["meta"] = metaRules 
    config["filter"] = filterRules
    config["select"] = selectRules 

    return config

def process_data(filePath):
    #Determine data set type and get config
    dataName = os.path.splitext(filePath.split('/')[-1])[0]
    configPath = "./data_config/" + dataName + ".cfg" 
    config = None

    if os.path.isfile(configPath):
        config = (parse_config(configPath))

    if config == None:
        return


    if "decals" in dataName.lower():
		#decals dataset
        pass	
    else:
		#manga dataset
        pass
    
	#Clean data
    #print("Cleaning data")
    df = (clean_data(config))

    #print("Selecting data")
    column_choices = (select_columns(df, config))
    
	#Retrieve column list for embedding choices

	#Return column list, cleaned and meta data
    return [config, df, column_choices]


def read_data(directory):

    dataPairs = []
    try:
        # Iterate over files in the directory
        for filename in os.listdir(directory):
            # Check if it's a file (not a subdirectory)
            if os.path.isfile(os.path.join(directory, filename)):
                pair = process_data(os.path.join(directory, filename))
                if pair != None:
                    dataPairs += [pair]
                #read_data(parse_config(os.path.join(directory, filename)))

    except OSError as e:
        print(f"Error reading directory: {e}")

    return dataPairs

#directory_path = './data'

#read_data(directory_path)