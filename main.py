from utils import *

def main():
    # Limit data to first 4999 entries from folder images_001
    df = load_data().head(4999)

    df = clean_dataframe(df)

    df_filt = filter_classes(df)
    df_analysis(df_filt)


    

    # Note: Not implemented yet
    model = make_model()
    x_train, y_train, x_test, y_test = get_train_test_split(df, 5)
    train_model(model, x_train, y_train)
    predict_labels(model, x_test, y_test)

if __name__ == '__main__':
    main()
    