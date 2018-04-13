import sqlite3

def main():
    output_file_name = 'MSD_track_id_and_year.txt'
    conn_match = sqlite3.connect('mxm_dataset.db')
    conn_MSD = sqlite3.connect('track_metadata.db')
    get_track_ids = 'SELECT DISTINCT track_id, mxm_tid FROM lyrics;'
    get_year = 'SELECT year FROM songs WHERE track_id =\'{}\';'
    output_string = '{}\t{}\n'
    # get all unique IDs
    output_file = open(output_file_name, 'w')
    all_track_ids = conn_match.execute(get_track_ids).fetchall()
    for id_row in all_track_ids:
        track_id = id_row[0]
        mxm_tid = id_row[1]
        get_year_from_id = get_year.format(track_id)
        year_row = conn_MSD.execute(get_year_from_id).fetchone()
        if year_row is not None:
            year = year_row[0]
            if year != 0:
                output_file.write(output_string.format(mxm_tid, year))
    output_file.close()

if __name__ == '__main__':
    main()
