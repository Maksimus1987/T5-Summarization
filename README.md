[![ru](https://img.shields.io/badge/lang-ru-red.svg)](/README.ru.md)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/italian/simple_automatic_machine_learning_pipeline/blob/main/LICENSE)

# Конвейер машинного обучения

Этот репозиторий содержит набор скриптов на Python и bash, которые используются для создания, предобработки, обучения и тестирования модели машинного обучения для предсказания температуры.

## Содержание репозитория

- `data_creation.py`: скрипт для создания наборов данных, описывающих изменение дневной температуры. Создает несколько наборов данных, некоторые из которых могут содержать аномалии или шумы. Сохраняет наборы данных в папках `train` и `test`.
- `model_preprocessing.py`: скрипт для предобработки данных. Выполняет нормализацию признаков с помощью `sklearn.preprocessing.StandardScaler`.
- `model_preparation.py`: скрипт для создания и обучения модели машинного обучения на данных из папки `train`.
- `model_testing.py`: скрипт для тестирования модели машинного обучения на данных из папки `test`. Вычисляет среднеквадратичное отклонение между предсказанными и фактическими значениями температуры.
- `pipeline.sh`: bash-скрипт, последовательно запускающий все python-скрипты.

## Требования

Для запуска скриптов необходимо установить следующие библиотеки Python:

- numpy
- pandas
- scikit-learn

## Инструкции по использованию

1. Скачайте или скопируйте репозиторий на свой компьютер.
2. Откройте терминал и перейдите в папку с репозиторием.
3. Запустите bash-скрипт `pipeline.sh` следующей командой:


Скрипты последовательно выполнят создание наборов данных, предобработку, обучение и тестирование модели машинного обучения. Результаты тестирования будут выведены в консоль.

## Лицензия

Этот проект выпущен под лицензией MIT. См. файл `LICENSE` для получения дополнительной информации.
