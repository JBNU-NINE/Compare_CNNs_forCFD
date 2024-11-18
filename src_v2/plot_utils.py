import matplotlib.pyplot as plt
import parser
import os
import torch
import matplotlib.animation as animation
import numpy as np
import residual
import copy


class PlotUtils:
    true_dict = {}
    pred_dict = {}
    def __init__(self, test_dir_name):
        self.__set_log_dir(test_dir_name)
        self.threshold_t = 0.4
        self.threshold_ux = 0.024
        self.threshold_uy = None
        self.threshold_dict = {
            "t": 0.4,
            "ux": 0.024,
            "uy": self.threshold_uy,
        }
        self.allowed_dtypes = ["t", "ux", "uy"]

    def __set_log_dir(self, test_dir_name):
        self.log_dir = parser.TrainingConfig().log_dir
        self.model_name = parser.ModelConfig().model_name
        self.plot_dir = os.path.join(self.log_dir, self.model_name, test_dir_name)
        if not os.path.exists(self.plot_dir):
            raise Exception("Log directory does not exist")
        self.plot_dir = os.path.join(self.plot_dir, "plots")

        if not os.path.exists(self.plot_dir):
            print("Directory does not exist, creating one")
            os.makedirs(self.plot_dir)

    def save_animation(self, img_arr, animation_path):
        fig = plt.figure()
        im = plt.imshow(img_arr[0])
        # save the entire numpy array as npz_compressed file 
        print(f"Saving compressed file too at:")
        np.savez_compressed(os.path.join(self.plot_dir, f"{animation_path}_animation.npz"), img_arr)

        def updatefig(j):
            im.set_array(img_arr[j])
            return [im]

        ani = animation.FuncAnimation(
            fig, updatefig, frames=len(img_arr), interval=50, blit=True
        )
        ani.save(
            os.path.join(self.plot_dir, f"{animation_path}_animation.gif"),
            writer="imagemagick",
            fps=60,
        )
        # Close the figure
        plt.close(fig)

    def __get_mae_arr(self, predicted_y, test_y):
        mae_arr = []
        for i in range(predicted_y.shape[0]):
            mae_arr.append(torch.mean(torch.abs(predicted_y[i] - test_y[i])).item())
        return mae_arr

    def __get_max_ae_arr(self, predicted_y, test_y):
        max_ae_arr = []
        for i in range(predicted_y.shape[0]):
            max_ae_arr.append(torch.max(torch.abs(predicted_y[i] - test_y[i])).item())
        return max_ae_arr

    def plot_mae(self, predicted_y, test_y, d_type):
        curr_d_type = d_type.split("_")[1]
        assert (
            curr_d_type in self.allowed_dtypes
        ), f"Data type: {d_type, curr_d_type} not allowed"
        mae_arr = self.__get_mae_arr(predicted_y, test_y)
        plt.figure()
        plt.plot(mae_arr)
        plt.savefig(os.path.join(self.plot_dir, f"{d_type}_mae.png"))

    def plot_max_ae(self, predicted_y, test_y, d_type):
        """
        I haven't changed this for same scale 
        because while testing the numbers can vary a lot.
        """
        curr_d_type = d_type.split("_")[1]
        assert (
            curr_d_type in self.allowed_dtypes
        ), f"Data type: {d_type, curr_d_type} not allowed"
        mae_arr = self.__get_max_ae_arr(predicted_y, test_y)
        mae_arr = np.array(mae_arr)
        threshold = self.threshold_dict[curr_d_type]
        plt.figure()
        plt.plot(mae_arr)
        if threshold is not None:
            threshold_index = np.where(mae_arr > threshold)
            if thres_cond := bool(np.any(threshold_index[0])):
                threshold_index = threshold_index[0][0]
                print(f"Threshold reached at {threshold_index} for data: {curr_d_type}")
                x = np.arange(mae_arr.shape[0])
                if thres_cond:
                    plt.axvline(
                        x=x[threshold_index],
                        color="red",
                        linestyle="--",
                        label=f"Threshold ({threshold}) {curr_d_type} at {threshold_index}",
                    )
                    # plt.axhline(
                    #     y=threshold,
                    #     color="purple",
                    #     linestyle="--",
                    #     label=f"Threshold exactly at:",
                    # )
                plt.legend()
        plt.xlabel("X")
        plt.ylabel("Error")
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, f"{d_type}_max_ae.png"))
        print(f"Saving Numpy array of all predictions")
        save_path_pred = os.path.join(self.plot_dir, f"{d_type}_predictions.npy")
        save_path_true = os.path.join(self.plot_dir, f"{d_type}_true.npy")
        np.save(save_path_pred, predicted_y)
        np.save(save_path_true, test_y)
        print(f"Saved Predictions")

    def plot_accuracy(self, predicted_y, test_y, d_type):
        d_type_ = d_type.split("_")[1]
        self.true_dict[d_type_] = np.squeeze(copy.deepcopy(test_y), axis = 1)
        self.pred_dict[d_type_] = np.squeeze(copy.deepcopy(predicted_y), axis = 1)
        self.plot_mae(predicted_y, test_y, d_type)
        self.plot_max_ae(predicted_y, test_y, d_type)
    
    def plot_residual(self, true_residual, pred_residual, residual_type):    
        plot_dir = os.path.join(self.plot_dir, f"residual_{residual_type}.svg")
        plt.figure()
        plt.plot(true_residual, label="True Residual", linewidth = 4.0)
        plt.plot(pred_residual, label="Predicted Residual")
        plt.ylabel("Residual Value")
        plt.xlabel("Timestep")
        plt.title(f"Residual Plot for {residual_type}")
        plt.grid()
        plt.legend()
        plt.savefig(plot_dir, format = "svg", dpi = 1200)
    def calculate_residual(self):
        true_ux_arr = self.true_dict["ux"]
        true_uy_arr = self.true_dict["uy"]
        true_t_arr = self.true_dict["t"]
        
        pred_ux_matrix = self.pred_dict["ux"]
        pred_uy_matrix = self.pred_dict["uy"]
        pred_t_matrix = self.pred_dict["t"]

        true_rs_mass_arr = []
        pred_rs_mass_arr = []
        true_rs_momentum_arr = []
        pred_rs_momentum_arr = []
        true_rs_heat_arr = []
        pred_rs_heat_arr = []
        print(f"Shape of pred_ux_matrix: {pred_ux_matrix.shape}")
        for i in range(1,pred_ux_matrix.shape[0]):
            Rs_mass_pred = residual.residual_mass(pred_ux_matrix[i], pred_uy_matrix[i])
            Rs_mass_true = residual.residual_mass(true_ux_arr[i], true_uy_arr[i])
            true_rs_mass_arr.append(Rs_mass_true)
            pred_rs_mass_arr.append(Rs_mass_pred)

            Rs_momentum_pred = residual.residual_momentum(pred_ux_matrix[i], pred_ux_matrix[i-1], pred_uy_matrix[i], pred_t_matrix[i])
            Rs_momentum_true = residual.residual_momentum(true_ux_arr[i], true_ux_arr[i-1], true_uy_arr[i], true_t_arr[i])
            true_rs_momentum_arr.append(Rs_momentum_true)
            pred_rs_momentum_arr.append(Rs_momentum_pred)

            Rs_heat_pred = residual.residual_heat(pred_ux_matrix[i], pred_uy_matrix[i], pred_t_matrix[i], pred_t_matrix[i-1])
            Rs_heat_true = residual.residual_heat(true_ux_arr[i], true_uy_arr[i], true_t_arr[i], true_t_arr[i-1])
            true_rs_heat_arr.append(Rs_heat_true)
            pred_rs_heat_arr.append(Rs_heat_pred)
        
        self.plot_residual(true_rs_mass_arr, pred_rs_mass_arr, "mass")
        self.plot_residual(true_rs_momentum_arr, pred_rs_momentum_arr, "momentum")
        self.plot_residual(true_rs_heat_arr, pred_rs_heat_arr, "heat")

if __name__ == "__main__":
    plotutils = PlotUtils()
