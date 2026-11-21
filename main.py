import utilities as u
from model import Model
import constants as c
from sklearn.preprocessing import MultiLabelBinarizer


def main():
    # Limit data to first 4999 entries from folder images_001
    df = u.load_data().head(4999)

    df = u.clean_dataframe(df)
    df_filt = u.filter_classes(df)
    u.df_analysis(df_filt)

    model = Model((206, 206))
    x_train, x_test, y_train, y_test = u.get_train_test_split(df_filt, 2, (206, 206))

    mlb = MultiLabelBinarizer(classes=c.CLASSES_LIST)
    y_train_lists = [s.split("|") if isinstance(s, str) and s else [] for s in y_train]
    y_train_bin = mlb.fit_transform(y_train_lists)

    model.train_model(x_train, y_train_bin)

    metrics = model.predict_multilabel(x_test, y_test, mlb, threshold=0.5)
    print(metrics)

if __name__ == '__main__':
    main()
    