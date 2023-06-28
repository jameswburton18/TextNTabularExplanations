for model in bert deberta disbert drob
do
    # scp imdb_genre_${exp}_shap_values.npy office:CodingProjects/TextNTabularExplanations/
    # scp imdb_genre_${exp}_val_shap_values.npy office:CodingProjects/TextNTabularExplanations/shap_vals/imdb_genre/
    scp -r models/shap_vals_${model}_sf1/ office:CodingProjects/TextNTabularExplanations/models/
done