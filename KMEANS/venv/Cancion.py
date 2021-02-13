
class Cancion:
    def __init__(self,id,nombre,artista,danceability, energy, loudness, speechiness, acousticness, instrumentalness,liveness, valence, tempo):
        self.nombre=nombre
        self.artista=artista
        self.danceability=danceability
        self.energy=energy
        self.loudness=loudness
        self.speechiness=speechiness
        self.acousticness=acousticness
        self.instrumentalness=instrumentalness
        self.liveness=liveness
        self.valence=valence
        self.tempo=tempo
        self.id=id
    def __repr__(self):
        return self.nombre+':'+self.artista
        
    