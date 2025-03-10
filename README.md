# Style-Transfer
Этот проект реализует алгоритм произвольного переноса стиля с использованием метода Adaptive Instance Normalization (AdaIN) на базе фреймворка PyTorch. Алгоритм позволяет переносить стиль с одного изображения на другое в режиме реального времени, предоставляя пользователю возможность создавать уникальные композиции. Для демонстрации работы алгоритма используется веб-интерфейс, разработанный на базе Streamlit.

Алгоритм основан на исследовании Huang, X. & Belongie, S. (2017) "Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization", представленном на конференции ICCV 2017. Метод AdaIN изменяет статистику признаков (среднее и стандартное отклонение) изображения-содержимого в соответствии с признаками изображения-стиля, что позволяет эффективно объединять содержание одного изображения и стиль другого.

git clone https://github.com/DashaRudas/Style-Transefer.git

cd Style-Transefer

pip install -r requirements.txt

streamlit run streamlit_app.py
