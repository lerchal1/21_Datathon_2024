import pandas as pd

df = pd.read_csv("data/skylab_instagram_datathon_dataset.csv", sep=";")
df_original = df.copy(deep=True)


df = df_original.copy(deep=True) 
df.drop(columns=["period", "calculation_type", "domicile_country_name","primary_exchange_name", "compset"], inplace=True)
df.sort_values(by=["period_end_date"], inplace=True, ascending=False)

df.drop_duplicates(inplace=True)
df.sort_values(by=["business_entity_doing_business_as_name", "period_end_date"], inplace=True, ascending=False)

#taking timedelta from first available end date instead of absolute date (COMMENT OUT AFTER FIRST RUN)
df['period_end_date'] = pd.to_datetime(df['period_end_date'])
min_date = df['period_end_date'].min()
df['period_end_date'] = df['period_end_date'] - min_date
#order by period_end_date
df.sort_values(by="period_end_date", inplace=True)

cathegories = ["followers", "likes", "comments", "videos", "pictures"]

#fill nan with zeros
for name in cathegories:
    df[name].fillna(0, inplace=True)

df.to_csv("data/cleaned_data.csv")