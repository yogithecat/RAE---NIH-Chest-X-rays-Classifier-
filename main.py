from utils import *

def main():
    df = load_data()

    df = clean_dataframe(df)

    df_filt = filter_classes(df)
    df_analysis(df_filt)


    

    # Note: Not implemented yet
        # make_model()
        # get_train_test_split()
        # train_model()
        # predict_labels()

if __name__ == '__main__':
    main()
    