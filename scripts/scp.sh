for exp in 0
do
    # scp imdb_genre_${exp}_shap_values.npy office:CodingProjects/TextNTabularExplanations/
    scp imdb_genre_${exp}_val_shap_values.npy office:CodingProjects/TextNTabularExplanations/shap_vals/imdb_genre/
done