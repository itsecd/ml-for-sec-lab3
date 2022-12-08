# Машинное обучение для задач информационной безопасности. Лаба 3. Порядок выполнения

## Схема сдачи

1. Сделать форк данного репозитория
2. Выбрать себе (с учётом указанных в описании задания ограничений) вариант задания и вписать его в табличку https://docs.google.com/spreadsheets/d/1Z1tym9FfX-Dj8huP2Q3iWVm0lrPpQz6NCGiweYGVy1w/edit?usp=sharing
3. Выполнить задание согласно выбранному варианту
4. Сделать pull request в данный репозиторий, содержащий один файл `.ipynb`.
5. Получить результат в рамках code review с замечаниями по коду.
6. При необходимости повторять пп. 3-4, пока преподаватель не отправит approve.
7. Во время онлайн-занятия защитить работу, ответить на вопросы преподавателя

## Более подробные рекомендации по работе с кодом

1. Форк *необходимо* сделать сразу. Для преподавателя это сигнализирует о том, что студент приступил к работе.
2. В описании репозитория нужно указать свои ФИО.
3. Желательно почаще делать коммиты. В идеале - как только решена некоторая промежуточная задача.
4. Коммиты *должны* иметь вменяемые описания.
5. Рекомендуется, чтобы ваш репозиторий содержал файлы [.gitignore](https://docs.github.com/en/get-started/getting-started-with-git/ignoring-files) (для них имеется набор [шаблонов](https://github.com/github/gitignore)) и [requirements.txt](https://www.jetbrains.com/help/pycharm/managing-dependencies.html#create-requirements)

## Запуск в Google Colaboratory

Перед запуском необходимо выполнить следующие команды 
```
!sudo apt-get install libmagickwand-dev
!pip install --no-cache-dir \
    opencv-python-headless==4.6.*\
    rawpy==0.17.* \
    pandas \
    Pillow==7.1.2 \
    scikit-image==0.16.2 \
    scipy==1.4.1 \
    tqdm \
    Wand
```

## Локальный запуске через Docker

Рекоммендуется запускать через WSL либо на Unix системах.

Команда для сборки образа:
```bash
docker build -f Dockerfile.cpu -t ml-for-sec-3 .
```

Поднять контейнер можно следующей командой:
```bash
docker run --rm -it -v $(pwd):/workspace -u $(id -u):$(id -g) ml-for-sec-3
```