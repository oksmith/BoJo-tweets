def get_date_from_file_loc(file_loc):
    return file_loc.split('tweets_')[-1].split('.json')[0]
