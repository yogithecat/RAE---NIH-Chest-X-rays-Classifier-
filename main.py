import utilities as u
from model import Model

def main():
    # Limit data to first 4999 entries from folder images_001
    df = u.load_data().head(4999)

    df = u.clean_dataframe(df)
    df_filt = u.filter_classes(df)
    u.df_analysis(df_filt)

    model = Model((206, 206))
    x_train, x_test, y_train, y_test = u.get_train_test_split(df_filt, 2, (206, 206))
    model.train_model(x_train, y_train)
    model.predict_labels(x_test, y_test)

if __name__ == '__main__':
    main()
    