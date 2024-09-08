from fireClfStack import fireClfStack
from sklearn.model_selection import train_test_split

def modelTraining(dataset):
    model = fireClfStack()

    # Разделение датасета на признаки и таргет
    X = dataset.drop(columns=['fire'])
    y = dataset['fire']

    # Разделение выборки на тренировочную и тестовую
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Обучение модели
    model.fit(X_train, y_train)

    # Формирование метрки на заданной тестовой выборке
    model.printMCC(X_test, y_test)

    # Формирование итогового файла
    model.predict(X_test, y_test).to_csv('./res.csv')
