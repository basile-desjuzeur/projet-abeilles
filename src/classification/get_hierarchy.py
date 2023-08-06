import pandas as pd
import os
import tqdm
import argparse



###### ARGUMENTS ######

argparser = argparse.ArgumentParser(
    description='Get the hierarchy of the species in the csv file')

argparser.add_argument(
    '-t',
    '--path_taxref',
    help='path to the csv file of the taxref database')

argparser.add_argument(
    '-w',
    '--path_whole_dataset_csv',
    help='path to a csv file with all the species we want to get the hierarchy of')


argparser.add_argument(
    '-o',
    '--path_output_csv',
    help='path to output the csv files in a folder')


###### FUNCTIONS ######

def _main_(path_taxref, path_whole_dataset_csv, path_output_csv):
    """
    Return the hierarchy of the species in the csv file

    Input:
        - path_taxref: path to the csv file of the taxref database 
            contains : ['LB_NOM','GENRE', 'FAMILLE', 'SOUS_FAMILLE', 'TRIBU']
        - path_whole_dataset_csv: path to a csv file with all the species we want to get the hierarchy of
            contains : ['label']
        - path_output_csv: path to output the csv files in a folder

    Output:
        - csv files with the hierarchy of the species in the csv file : hierarchy.csv
        - csv files with the species that are not in taxref : not_in_taxref.csv
        - csv files with the species that are not in the form 'genus specie' : not_as_specie.csv
            (ex : Bombus instead of Bombus terrestris)
    """

    # Transform to dataframes
    df_taxref = pd.read_csv(path_taxref, sep=';', encoding_errors='ignore', low_memory=False)
    df_whole_dataset = pd.read_csv(path_whole_dataset_csv)

    # Get the species in datasets
    species_in_dataset = df_whole_dataset['label'].unique()

    not_in_taxref = []
    not_as_specie = []
    df_specie = pd.DataFrame()

    print('\nGetting the hierarchy of each specie...\n')

    # Get the hierarchy of each specie
    for specie in tqdm.tqdm(species_in_dataset):

        # add to nos_as_specie if is in one word (eg : Bombus)
        if len(specie.split(' ')) == 1:
            not_as_specie.append(specie)

        # Get the specie in taxref
        else :

            # Get the specie in taxref
            df_temp = df_taxref[df_taxref['LB_NOM'] == specie]


            # If the specie is not in taxref
            if df_temp.empty:
                not_in_taxref.append(specie)

            # If the specie is in taxref, concatenate the dataframes
            else:
                df_temp = df_temp[['LB_NOM', 'FAMILLE', 'SOUS_FAMILLE', 'TRIBU']]
                
                # genus is the first word of the specie
                genus = specie.split(' ')[0]
                df_temp['GENRE'] = genus

                # reset order of columns
                df_temp = df_temp[['LB_NOM','GENRE', 'FAMILLE', 'SOUS_FAMILLE', 'TRIBU']]

                # rename columns
                df_temp.columns = ['species', 'genus', 'family', 'subfamily', 'tribe']

                df_specie = pd.concat([df_specie, df_temp])

    smiley_check = u'\u2705'
    print('Done ', smiley_check)


    # Save the csvs

    print('Saving csvs...\n')

    df_not_in_taxref = pd.DataFrame(not_in_taxref)
    if df_not_in_taxref.empty:
        print('All the species are in taxref')
    else:
        df_not_in_taxref = df_not_in_taxref.sort_values(by=[0])
        df_not_in_taxref.to_csv(os.path.join(path_output_csv, 'not_in_taxref.csv'), index=False, header=False)
        print('Some species are not in taxref, see not_in_taxref.csv')


    df_not_as_specie = pd.DataFrame(not_as_specie)
    if df_not_as_specie.empty:
        print('All the species are in the form "genus specie"')
    else:
        df_not_as_specie = df_not_as_specie.sort_values(by=[0])
        df_not_as_specie.to_csv(os.path.join(path_output_csv, 'not_as_specie.csv'), index=False, header=False)
        print('Some species are not in the form "genus specie", see not_as_specie.csv')

    df_specie = df_specie.sort_values(by=['species'])
    df_specie.to_csv(os.path.join(path_output_csv, 'hierarchy.csv'), index=False)

    print('Done ' + smiley_check)

    # Print the summary

    print('-'*50)
    print('\t SUMMARY \t')
    print('Number of species in dataset : ', len(species_in_dataset))
    print('Number of species in taxref : ', len(df_specie))
    print('Number of species not in taxref : ', len(not_in_taxref))
    print('Number of species not in the form "Apis mellifera" : ', len(not_as_specie))
    print('-'*50)





if __name__ == '__main__':

    args = argparser.parse_args()

    _main_(args.path_taxref, args.path_whole_dataset_csv, args.path_output_csv)