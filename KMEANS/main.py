
from Spotify import SpotifyPro
from kmeans import Kmeans
# from spotipy import Spotify


def main():

    # k.importarDatos('')
    spotify=SpotifyPro()
    df=spotify.iniciar('1QP6tyANnZZ9bRTfQG4X7a')
    # print(df.head(1))
    k = Kmeans(df)
    if(len(df)):
        k.importarDatos()
    # k=Kmeans2(df)
    # k.iniciar()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
