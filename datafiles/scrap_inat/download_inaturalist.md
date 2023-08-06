

#  Téléchargement des métadonnées iNaturalist

Documentation : 

[https://github.com/inaturalist/inaturalist-open-data/tree/documentation](https://github.com/inaturalist/inaturalist-open-data/tree/documentation)

- Installer le client aws:

```sh

pip3 install --upgrade awscli
```

- Télécharger les csv photos, observations et taxons :

```sh

aws s3 cp --no-sign-request s3://inaturalist-open-data/photos.csv.gz photos.csv.gz
aws s3 cp --no-sign-request s3://inaturalist-open-data/observations.csv.gz observations.csv.gz
aws s3 cp --no-sign-request s3://inaturalist-open-data/taxa.csv.gz taxa.csv.gz
```

- Dézipper les fichiers : 

```sh

zip -r photos.csv.gz photos.csv
zip -r observations.csv.gz observations.csv
zip -r taxa.csv.gz taxa.csv
```

- S'il est prévu de mettre à jour régulièrement cette base de données, il peut être judicieux de renommer les csv avec la date correspondante :

```sh

mv photos.csv.gz photos_2305.csv
mv observations.csv.gz observations_2305.csv
mv taxa.csv.gz taxa_2305.csv
```

- Supprimer les zip :

```sh

rm -r photos.csv.gz
rm -r observations.csv.gz
rm -r taxa.csv.gz
```

**NB** : Le dossier doit aussi avoir un csv desired_taxa.csv avec une liste de noms latins

# Passage dans une base de données sqlite 

(Extrait et adapté de : https://forum.inaturalist.org/t/getting-the-inaturalist-aws-open-data-metadata-files-and-working-with-them-in-a-database/22135)



- Installation de SQLite :

```sh

sudo apt-get install sqlite3
```

- Création d'une nouvelle base de données : 

```sh

touch inat.db
sqlite3 inat.db
```

- Création des tables : 

```sql

CREATE TABLE observations (
    observation_uuid uuid NOT NULL,
    observer_id integer,
    latitude numeric(15,10),
    longitude numeric(15,10),
    positional_accuracy integer,
    taxon_id integer,
    quality_grade character varying(255),
    observed_on date
);

CREATE TABLE photos (
    photo_uuid uuid NOT NULL,
    photo_id integer NOT NULL,
    observation_uuid uuid NOT NULL,
    observer_id integer,
    extension character varying(5),
    license character varying(255),
    width smallint,
    height smallint,
    position smallint
);

CREATE TABLE taxa (
    taxon_id integer NOT NULL,
    ancestry character varying(255),
    rank_level double precision,
    rank character varying(255),
    name character varying(255),
    active boolean
);

CREATE TABLE observers (
    observer_id integer NOT NULL,
    login character varying(255),
    name character varying(255)
);


CREATE TABLE desired_taxa (
    name character varying(255)
);

```

- Pour vérifier que tout va bien, regarder les tables et leur contenu:

```sql 

.tables

.schema nom_table
```


- Mettre en place une importation de csv avec colonnes séparées par des tabulations :

```sql

.import taxa.csv taxa
.import observations.csv observations
.import photos.csv photos
.import desired_taxa.csv desired_taxa
```
.mode tabs

- Vérifier le bon import des données:

```sql

select * from taxa limit 10;
```

- Création d'indexes pour accélérer les requêtes : 


```sql

.mode tabs

CREATE UNIQUE INDEX "idx_observations_observation_uuid" ON "observations" ("observation_uuid");
CREATE INDEX "idx_observations_observer_id" ON "observations" ("observer_id");
CREATE INDEX "idx_observations_taxon_id" ON "observations" ("taxon_id");
CREATE INDEX "idx_observations_quality_grade" ON "observations" ("quality_grade");
CREATE INDEX "idx_observations_observed_on" ON "observations" ("observed_on");
CREATE INDEX "idx_observations_longitude" ON "observations" ("longitude");
CREATE INDEX "idx_observations_latitude" ON "observations" ("latitude");

CREATE INDEX "idx_photos_photo_uuid" ON "photos" ("photo_uuid");
CREATE INDEX "idx_photos_observation_uuid" ON "photos" ("observation_uuid");
CREATE INDEX "idx_photos_photo_id" ON "photos" ("photo_id");
CREATE INDEX "idx_photos_observer_id" ON "photos" ("observer_id");
CREATE INDEX "idx_photos_license" ON "photos" ("license");

CREATE UNIQUE INDEX "idx_taxa_taxon_id" ON "taxa" ("taxon_id");
CREATE INDEX "idx_taxa_name" ON "taxa" ("name");
CREATE INDEX "idx_taxa_rank" ON "taxa" ("rank");
CREATE INDEX "idx_taxa_rank_level" ON "taxa" ("rank_level");
CREATE INDEX "idx_taxa_ancestry" ON "taxa" ("ancestry");

CREATE UNIQUE INDEX "idx_desired_taxa_name" ON "desired_taxa" ("name");
```

- Pour vérifier que tout a bien été créé : 

```sql

.indices
```

**Le fichier de BD fait 44Go à la fin de tout ça ! (19/04/2022)**



# Requêtes 


## Filtre géographique : uniquement les photos prises en France (optionnel)


- On approxime la surface de la France par un cercle de rayon 500km partant de Limoges (lat,long : 45.85,1.86).

Voir [ici](../images/perimeter.png).

- On postule que la surface est plane et que le rayon de la terre en France vaut 6371 km et on a : 

distance(point_a,point_b) = acos(sin(lat1) * sin(lat2)+cos(lat1)* cos(lat2) * cos(lon2-lon1)) * 6371


```sql

ALTER TABLE observations ADD COLUMN in_circle BOOLEAN;

UPDATE observations SET in_circle = 
    CASE 
        WHEN (
            ACOS(SIN(RADIANS(45.85)) * SIN(RADIANS(latitude)) +
                 COS(RADIANS(45.85)) * COS(RADIANS(latitude)) *
                 COS(RADIANS(1.86 - longitude)))
            ) * 6371 <= 500 THEN 1
        ELSE 0
    END;


.mode tabs 
CREATE INDEX "idx_observations_in_circle" ON "observations" ("in_circle");
```

## Filtre qualitatif : uniquement les photos validées par des chercheurs

```sql
CREATE TABLE inat_filter (
    name character varying(255),
    taxon_id integer,
    photo_id integer,
    extension character varying(5),
    observation_uuid uuid,
    quality_grade character varying(255),
    latitude numeric(15,10),
    longitude numeric(15,10),
    in_circle boolean
);


INSERT INTO inat_filter (name, taxon_id, photo_id, extension, observation_uuid, quality_grade, latitude, longitude, in_circle)
SELECT dt.name, tx.taxon_id, ph.photo_id, ph.extension, ob.observation_uuid, ob.quality_grade, ob.latitude, ob.longitude, ob.in_circle
FROM desired_taxa dt
JOIN taxa tx ON dt.name = tx.name
JOIN observations ob ON tx.taxon_id = ob.taxon_id
JOIN photos ph ON ob.observation_uuid = ph.observation_uuid
WHERE ob.quality_grade = 'research' AND ob.in_circle = 1;
```

NB : Enlever tout ou partie de la dernière ligne si nécessaire.

# Edition du csv de photos à télécharger

```sql
.headers on
.mode csv
.output inat_filter.csv
SELECT name,taxon_id, photo_id, extension FROM inat_filter;

.output stdout
```

(Adapté du guide d'Axel Carlier)

