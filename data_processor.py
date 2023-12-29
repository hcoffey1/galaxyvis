import os
import shlex
import pandas as pd

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
           #print(hdu)
           df_list += [read_fits(filePath, hdu)]

       df = pd.concat(df_list, axis=1)

       #df = read_fits(filePath)

    #Set columns to lowercase
    #df.rename(columns={col: col.lower() for col in df.columns}, inplace=True)

    for rule in config["filter"]:

        if "==" in rule:
            index = rule.index("==")
            df = df[df[rule[index-1]] == float(rule[index+1])]

        if "CONTAINS" in rule:
            term = (rule[rule.index("CONTAINS") + 1])

            if "RANGE" in rule:
                range = (rule[rule.index("RANGE") + 1])
                range = range.split(',')

                filtered_columns = [col for col in df.columns if term in col]
                for col in filtered_columns:
                    df = df[df[col] >= float(range[0])] 
                    df = df[df[col] <= float(range[1])] 
        
        if "DROP" in rule:
            if "NA" in rule:
                df = df.dropna(axis=1)
        
        if "ALL" in rule:
            index = rule.index("ALL")
            rows_to_remove = df.map(lambda x: isinstance(x, (int, float)) and x < float(rule[index+2])).any(axis=1)
            df = df[~rows_to_remove] #filter out errors from firefly


    return df

def select_columns(df, config):

    columns = []
    for rule in config["select"]:
        #print(rule)

        if "CONTAINS" in rule:
            columns += ([col for col in df.columns if rule[1] in col ])#Galaxy Zoo]

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
                tokens = shlex.split(line)

                if tokens[0] == "TYPE":
                    config['type'] = tokens[1]
                elif tokens[0] == "HDU":
                    config['hdu'] = ([int(x) for x in tokens[1].split(',')])
                elif tokens[0] == "LABEL":
                    config['label'] = tokens[1]
                elif tokens[0] == "META":
                    metaPhase = True
                elif metaPhase:
                    if tokens[0] == "FILTER":
                        metaPhase = False 
                        filtering = True
                        continue
                    metaRules.append(tokens)
                elif filtering:
                    if tokens[0] == "SELECT":
                        filtering = False
                        selecting = True
                        continue
                    filterRules.append(tokens)
                elif selecting:
                    selectRules.append(tokens)

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