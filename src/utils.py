import numpy as np

def row_to_string(row, cols):
    row['text'] = ' | '.join(f'{col}: {row[col]}' for col in cols)
    return row

def multiple_row_to_string(row, cols, multiplier=1, nodesc=False):
    row['text'] = ' | '.join(f'{col}: {(str(row[col]) + " ")*multiplier}' for col in cols)
    if not nodesc:  
        row['text'] = row['text'] + ' | Description: ' + row['Description']
    return row

def prepare_text(dataset, version):
    if version == 'all_text':
        cols = ['Year','Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)','Metascore', 'Rank', 'Description']
        dataset = dataset.map(row_to_string, fn_kwargs={'cols': cols})
        return dataset
    elif version == 'tab_as_text':
        cols = ['Year','Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)','Metascore', 'Rank']
        dataset = dataset.map(row_to_string, fn_kwargs={'cols': cols})
        return dataset
    elif version == 'tabx5':
        cols = ['Year','Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)','Metascore', 'Rank']
        dataset = dataset.map(multiple_row_to_string, fn_kwargs={'cols': cols, 'multiplier': 5})
        return dataset
    elif version == 'tabx5_nodesc':
        cols = ['Year','Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)','Metascore', 'Rank']
        dataset = dataset.map(multiple_row_to_string, fn_kwargs={'cols': cols, 'multiplier': 5, 'nodesc': True})
        return dataset
    elif version == 'tabx2_nodesc':
        cols = ['Year','Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)','Metascore', 'Rank']
        dataset = dataset.map(multiple_row_to_string, fn_kwargs={'cols': cols, 'multiplier': 2, 'nodesc': True})
        return dataset
    elif version == 'tabx2':
        cols = ['Year','Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)','Metascore', 'Rank']
        dataset = dataset.map(multiple_row_to_string, fn_kwargs={'cols': cols, 'multiplier': 2})
        return dataset
    elif version == 'reorder1':
        cols = ['Votes', 'Revenue (Millions)', 'Metascore', 'Rank', 'Description','Year','Runtime (Minutes)', 'Rating']
        dataset = dataset.map(row_to_string, fn_kwargs={'cols': cols})
        return dataset
    elif version == 'reorder2':
        cols = ['Description', 'Rank', 'Metascore', 'Revenue (Millions)', 'Votes', 'Rating', 'Runtime (Minutes)', 'Year']
        dataset = dataset.map(row_to_string, fn_kwargs={'cols': cols})
        return dataset
    elif version == None:
        # dataset rename column
        dataset = dataset.rename_column('Description', 'text')
        return dataset
    else:
        raise ValueError(f'Unknown version: {version}')
    
def format_text_pred(pred):
    if pred['label'] == 'LABEL_1':
        return np.array([1-pred['score'], pred['score']])
    else:
        return np.array([pred['score'], 1-pred['score']])
    
def select_prepare_array_fn(model_name):
    if model_name == 'imdb_genre_1': # tab as text
        cols = ['Year','Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)','Metascore', 'Rank']
        def array_fn(array):
            return np.array(
                " | ".join([f"{col}: {val}" for col, val in zip(cols, array)]), dtype="<U512"
                )
    elif model_name == 'imdb_genre_6': # tabx2_nodesc
        cols = ['Year','Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)','Metascore', 'Rank']
        def array_fn(array):
            return np.array(
                ' | '.join([f'{col}: {(str(val) + " ")*2}' for col, val in zip(cols, array)]), dtype="<U512"
                )
    elif model_name == 'imdb_genre_5': # tabx5_nodesc
        cols = ['Year','Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)','Metascore', 'Rank']
        def array_fn(array):
            return np.array(
                ' | '.join([f'{col}: {(str(val) + " ")*5}' for col, val in zip(cols, array)]), dtype="<U512"
                )
    elif model_name == 'imdb_genre_7': # tabx2
        cols = ['Year','Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)','Metascore', 'Rank']
        def array_fn(array):
            return np.array(
                ' | '.join([f'{col}: {(str(val) + " ")*2}' for col, val in zip(cols, array[:-1])])
                + ' | Description: ' + array[-1], dtype="<U512"
                )
    elif model_name == 'imdb_genre_2': # tabx5
        cols = ['Year','Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)','Metascore', 'Rank']
        def array_fn(array):
            return np.array(
                ' | '.join([f'{col}: {(str(val) + " ")*5}' for col, val in zip(cols, array[:-1])])
                + ' | Description: ' + array[-1], dtype="<U512"
                )
    elif model_name == 'imdb_genre_3': # reorder1
        cols = ['Votes', 'Revenue (Millions)', 'Metascore', 'Rank', 'Description', 'Year', 'Runtime (Minutes)', 'Rating']
        def array_fn(array):
            return np.array(
                " | ".join([f"{col}: {val}" for col, val in zip(cols, array)]), dtype="<U512"
                )
    elif model_name == 'imdb_genre_4': # reorder2
        cols = ['Description', 'Rank', 'Metascore', 'Revenue (Millions)', 'Votes', 'Rating', 'Runtime (Minutes)', 'Year']
        def array_fn(array):
            return np.array(
                " | ".join([f"{col}: {val}" for col, val in zip(cols, array)]), dtype="<U512"
                )
    
    return array_fn
        