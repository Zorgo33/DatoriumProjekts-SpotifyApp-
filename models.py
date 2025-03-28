from peewee import *
import datetime

db = SqliteDatabase('spotify.db')

class Track(Model):
    track_id = CharField(unique=True)  # Ensure track_id is unique
    track_name = CharField()
    track_artist = CharField(null=True)
    track_popularity = IntegerField()
    track_album_id = CharField(null=True)
    track_album_name = CharField(null=True)
    track_album_release_date = DateField()
    playlist_name = CharField(null=True)
    playlist_id = CharField(null=True)
    playlist_genre = CharField()
    playlist_subgenre = CharField(null=True)
    danceability = FloatField()
    energy = FloatField()
    key = IntegerField(null=True)
    loudness = FloatField(null=True)
    mode = IntegerField(null=True)
    speechiness = FloatField(null=True)
    acousticness = FloatField(null=True)
    instrumentalness = FloatField(null=True)
    liveness = FloatField(null=True)
    valence = FloatField(null=True)
    tempo = FloatField()
    duration_ms = IntegerField()
    created_at = DateTimeField(default=datetime.datetime.now)  # Track when the record is added

    class Meta:
        database = db
        table_name = 'track'

# Create the table if it doesn't exist
with db:
    db.create_tables([Track])
