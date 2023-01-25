# filmweb_recommendation_system



Dobra trochę pozmieniałem strukturę. (tylko src, reszta bez zmian)

Wszystkie modele wyjałem z jupytera i przeniosłem do folderu model do odpowiednich plików.
Musiałem to trochę pozmieniac.
    - usunąłem zmienne globalne;
    - ujednoliciłem interfejs;
    - dodałem zwracanie score przez każdy model
    - jeżeli się wywoła model z num_of_recomendations=-1 poda wszystkie.
  
To co dodałem nowego to połączony model i fitowanie modelu przy użyciu biblioteki optuna
plik fit. 
sprawdzenie rezultatów jest w
show_fitting.ipynb

