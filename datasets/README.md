# Datasets description

* `pokemon` - loaded from [Kaggle The Complete Pokemon Dataset](https://www.kaggle.com/datasets/rounakbanik/pokemon)

* `crime_classification` - loaded from [Kaggle Crime Classification](https://www.kaggle.com/datasets/yasiradnan/crime-classifcication)

* `south_korean_pollution` - loaded from [Kaggle South Korean Pollution](https://www.kaggle.com/datasets/calebreigada/south-korean-pollution?select=south-korean-pollution-data.csv)

* `human_faces` - loaded from [Kaggle Human Faces](https://www.kaggle.com/datasets/ashwingupta3012/human-faces)

---

<br>

## Detailed information

<br>

> ### Pokemon

<br>
This dataset contains information on all 802 Pokemon from all Seven Generations of Pokemon.

<br>

* `name`: The English name of the Pokemon
* `japanese_name`: The Original Japanese name of the Pokemon
* `pokedex_number`: The entry number of the Pokemon in the National Pokedex
* `percentage_male`: The percentage of the species that are male. Blank if the Pokemon is genderless.
* `type1`: The Primary Type of the Pokemon
* `type2`: The Secondary Type of the Pokemon
* `classification`: The Classification of the Pokemon as described by the Sun and Moon Pokedex
* `height_m`: Height of the Pokemon in metres
* `weight_kg`: The Weight of the Pokemon in kilograms
* `capture_rate`: Capture Rate of the Pokemon
* `baseeggsteps`: The number of steps required to hatch an egg of the Pokemon
abilities: A stringified list of abilities that the Pokemon is capable of having
* `experience_growth`: The Experience Growth of the Pokemon
* `base_happiness`: Base Happiness of the Pokemon
* `against_?`: Eighteen features that denote the amount of damage taken against an attack of a particular type
* `hp`: The Base HP of the Pokemon
* `attack`: The Base Attack of the Pokemon
* `defense`: The Base Defense of the Pokemon
* `sp_attack`: The Base Special Attack of the Pokemon
* `sp_defense`: The Base Special Defense of the Pokemon
* `speed`: The Base Speed of the Pokemon
* `generation`: The numbered generation which the Pokemon was first introduced
* `is_legendary`: Denotes if the Pokemon is legendary.

<br>

> ### Crime Classification

<br>
Classification of crime dataset.

<br>

* `Dates`
* `Category`
* `Descript`
* `DayOfWeek`
* `PdDistrict`
* `Resolution`
* `Address`
* `X`
* `Y`

<br>

> ### South Korean Pollution

<br>
Air pollution is a major problem in South Korea. On days with high pollution, citizens are advised not to go outdoors. This is especially true for those who are elderly or have pre-existing medical conditions. Pollution levels are higher at certain times of year and can change rapidly based on meteorological effects. Being able to accurately forecast the level of pollution would allow South Koreans to plan ahead and avoid exposing themselves to the harsh pollutants.

<br>

**Pollution data**

* `date` - date of measurement
* `pm25` - fine particulate matter (PM2.5) (µg/m3)
* `pm10` - fine particulate matter (PM10) (µg/m3)
* `o3` - Ozone (O3) (µg/m3)
* `no2` - Nitrogen Dioxide (NO2) (ppm)
* `so2` - Sulfur Dioxide (SO2) (ppm)
* `co` - Carbon Monoxide (CO) (ppm)
* `Lat` - Latitude where measurement was taken
* `Long` - Longitude where measurement was taken
* `City` - City where measurement was taken
* `District` - District where measurement was taken
* `Country` - Country where measurement was taken

**Weather Data (auxiliary)**

* `STATION` - Station Number
* `NAME` - Station Name
* `LATITUDE` - Latitude of station
* `LONGITUDE` - Longitude of station
* `ELEVATION` - Elevation of station
* `DATE` - Date of observation
* `LIQUID_PRECIPITATION` - Liquid precipitation (AA1)
* `SNOW_DEPTH` - Snow depth (AJ1)
* `DEW` - Dew
* `EXTREME_AIR_TEMP` - Extreme air temperature (KA1)
* `ATMOSPHERIC_PRESSURE` - Atmospheric pressure (MA1)
* `SEA_LEVEL_PRESSURE` - Sea level pressure (SLP)
* `TEMP` - Temperature (TMP)
* `VIS` - Visibility
* `WND` - Wind

For more detailed information about each field, you can view the documentation here: (documentation)[https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf]. NOTE: some field names were changed for clarity -- if so, original field names are in parenthesis.

<br>

> ### Human Faces

<br>
A collection of 7.2k+ images useful for multiple use cases such image identifiers, classifier algorithms etc. A thorough mix of all common creeds, races, age groups and profiles.