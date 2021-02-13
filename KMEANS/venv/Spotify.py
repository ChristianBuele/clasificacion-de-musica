import spotipy
from spotipy import SpotifyClientCredentials, util
from Cancion import  Cancion
import pandas as pd
import numpy as np
class SpotifyPro:
    client_id = 'bf4d5b85f6c946e9b99f0fccfe4c1c6e'
    client_secret = '3238226e21004352acdb7ad2c9a4e357'
    redirect_uri = 'http://localhost:8081'
    username = '31d4craw673kyd2qycdsn3v3l6ru'
    scope = 'user-library-read'
    token=None
    spotify=None
    def iniciar(self,idPlaylist):
        token=util.prompt_for_user_token(username=self.username,scope=self.scope,client_id=self.client_id,client_secret=self.client_secret,redirect_uri=self.redirect_uri)
        if token:
            self.spotify=spotipy.Spotify(auth=token)
            resultado=self.spotify.playlist(playlist_id=idPlaylist)['tracks']
            if resultado:
                return self.crearDatos(resultado)
            return []
        return []
    def crearDatos(self,resultado):
        cancionesSalida=[]
        indices = ['cod', 'artista', 'nombre', 'tempo', 'energy', 'loudness', 'danceability', 'valence', 'acousticness',
                   'instrumentalness', 'liveness', 'speechiness']
        lista=[]
        for canciones in resultado['items']:
            cancion = canciones['track']
            x = cancion['artists'][0]
            artista = str(x['name'])
            cod = str(cancion['id'])
            nombreCancion = str(cancion['name'])
            acou_data = self.spotify.audio_features(cod)
            tempo = float(acou_data[0]['tempo'])
            energy = float(acou_data[0]['energy'])
            loudness = float(acou_data[0]['loudness'])
            danceability = float(acou_data[0]['danceability'])
            valence = float(acou_data[0]['valence'])
            acousticness = float(acou_data[0]['acousticness'])
            instrumentalness = float(acou_data[0]['instrumentalness'])
            liveness = float(acou_data[0]['liveness'])
            speechiness = float(acou_data[0]['speechiness'])
            # nuevaCancion=Cancion(id=id,artista=artista,nombre=nombreCancion,tempo=tempo,energy=energy,loudness=loudness,danceability=danceability,valence=valence,acousticness=acousticness,instrumentalness=instrumentalness,liveness=liveness,speechiness=speechiness)
            # cancionesSalida.append(nuevaCancion)
            #para las series
            temp=[cod,artista,nombreCancion,tempo,energy,loudness,danceability,valence,acousticness,instrumentalness,liveness,speechiness]
            lista.append(temp)
        # print(len(lista),':',len(indices))
        df=pd.DataFrame(np.array(lista),columns=indices)

        print(df.head())


        return df