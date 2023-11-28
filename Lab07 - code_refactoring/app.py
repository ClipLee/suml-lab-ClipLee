
from task1 import predict_y
from task2 import save_data_and_train_model


def main():
    while True:
        print("\nChoose function to launch:")
        print("1. predict_y")
        print("2. save_data_and_train_model")
        print("3. Exit")

        wybor = input("\nChoose a menu option: ")

        if wybor == "1":
            x_f1 = input("Input x: ")
            if x_f1:
                print(predict_y(x_f1))
            else:
                print("\n------value of x is required------\n")

        elif wybor == "2":
            print("Input x, y, model path and file name:")
            x_f2 = input("Input x: ")
            y_f2 = input("Input y: ")
            if x_f2 and y_f2:
                m_path = input("Input path to a model (Enter = skip): ")
                f_name = input("Input file name (Enter = skip): \n")
                save_data_and_train_model(
                    x_f2, y_f2, m_path or None, f_name or None)

            else:
                print("\n------x and y values are required.------\n")
        elif wybor == "3":
            break
        else:
            print("\n------Unknown choice. Try again.------\n")

if __name__ == "__main__":
    main()
