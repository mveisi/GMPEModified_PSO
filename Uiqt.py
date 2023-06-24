import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QGroupBox, QLabel, QLineEdit, QRadioButton, \
    QPushButton, QFileDialog, QButtonGroup
from PyQt5.QtCore import pyqtSignal
import os
import subprocess

# Set the command to run
command = "which matlab"

# Run the command and capture the output
matlab_run = subprocess.check_output(command, shell=True)

# Decode the output from bytes to string
matlab_run = matlab_run.decode("utf-8").split()[0]

# command = "which python"
# # Run the command and capture the output
# python_run = subprocess.check_output(command, shell=True)

# # Decode the output from bytes to string
# python_run = python_run.decode("utf-8").split()[0]

def isfloat(inp_val):
    try:
        return(float(inp_val))
    except ValueError:
        return(False)
class MyUI(QWidget):
    # create a custom signal
    close_signal = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.close_signal.connect(self.close)

    def init_ui(self):
        self.setWindowTitle("Modified PSO")
        layout = QVBoxLayout()
        self.resize(600, 400)

        # Input file field
        file_label = QLabel("Input File:")
        self.file_line_edit = QLineEdit()
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_file)
        file_layout = QVBoxLayout()
        file_layout.addWidget(self.file_line_edit)
        file_layout.addWidget(browse_button)
        file_group_box = QGroupBox()
        file_group_box.setLayout(file_layout)
        
        
        # number field in front of radio
        number_label_radio1 = QLabel("Synthetic error:")
        self.number_label_edit_radio1 = QLineEdit()
        self.number_label_edit_radio1.setEnabled(False)
        
        number_label_radio3 = QLabel("cutting for distance less than:")
        self.number_label_edit_radio3 = QLineEdit()
        self.number_label_edit_radio3.setEnabled(False)
        
        # Radio buttons
        radio_group_box = QGroupBox("Radio Buttons")
        radio_layout = QVBoxLayout()
        self.radio_button1 = QRadioButton("Synthetic Run")
        self.radio_button2 = QRadioButton("Real dataset Run")
        self.radio_button3 = QRadioButton("Cutting distances less than:")
        
        radio_button_group = QButtonGroup(self)
        radio_button_group.addButton(self.radio_button1)
        radio_button_group.addButton(self.radio_button2)
        radio_button_group.setExclusive(True)
        
        
        self.radio_button1.toggled.connect(self.toggle_radio1_input_line)
        self.radio_button3.toggled.connect(self.toggle_radio2_input_line)
        radio_layout.addWidget(self.radio_button1)
        radio_layout.addWidget(number_label_radio1)
        radio_layout.addWidget(self.number_label_edit_radio1)
        radio_layout.addWidget(self.radio_button2)
        radio_layout.addWidget(self.radio_button3)
        radio_layout.addWidget(self.number_label_edit_radio3)
        radio_group_box.setLayout(radio_layout)
        
        
        
        # Number fields
        number_label1 = QLabel("Number of Subspaces:")
        self.number_line_edit1 = QLineEdit()

        number_label2 = QLabel("Kink distance:")
        self.number_line_edit2 = QLineEdit()

       

        # Output folder field
        output_label = QLabel("Output Folder:")
        self.output_line_edit = QLineEdit()

        # Run button
        run_button = QPushButton("Run")
        run_button.clicked.connect(self.run_button_clicked)

        # Add all the widgets to the layout
        layout.addWidget(file_label)
        layout.addWidget(file_group_box)
        layout.addWidget(radio_group_box)
        layout.addWidget(number_label1)
        layout.addWidget(self.number_line_edit1)
        layout.addWidget(number_label2)
        layout.addWidget(self.number_line_edit2)
        layout.addWidget(output_label)
        layout.addWidget(self.output_line_edit)
        layout.addWidget(run_button)

        self.setLayout(layout)
        self.show()
    def toggle_radio1_input_line(self, checked):
        self.number_label_edit_radio1.setEnabled(checked)
    def toggle_radio2_input_line(self, checked):
        self.number_label_edit_radio3.setEnabled(checked)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input File")
        self.file_line_edit.setText(file_path)

    def run_button_clicked(self):
        input_file = self.file_line_edit.text()
        if (os.path.isfile(input_file)):
            input_file = input_file
        else:
            raise ValueError('File not found')
        if (self.radio_button1.isChecked()):
            Synthetic_run = True 
        else:
            Synthetic_run = False
            real_data_run = True 
        if (self.radio_button2.isChecked()):
            real_data_run = True 
            Synthetic_run = False
        else:
            Synthetic_run = True 
            real_data_run = False
        if (input_file == ""):
            input_file = os.path.join(os.getcwd(), "data_NWIran.dat")
        synthetic_error = self.number_label_edit_radio1.text()
        if (synthetic_error == ""):
            synthetic_error = 20.0
        else:
            if (isfloat(synthetic_error)):
                synthetic_error = isfloat(synthetic_error)
            else:
                raise ValueError('Invalid Synthetic error')
        N_subspace = self.number_line_edit1.text()
        if (N_subspace == ""):
            N_subspace = 3
        else:
            if (isfloat(N_subspace)):
                N_subspace = isfloat(N_subspace)
            else:
                raise ValueError('Invalid Number of subspaces')
        kink_val = self.number_line_edit2.text()
        if (kink_val == ""):
            kink_val = 3600
        else:
            if(isfloat(kink_val)):
                kink_val = isfloat(kink_val)
            else:
                raise ValueError('Invalid Kink distance')
        cut_val = self.number_label_edit_radio3.text()
        if (cut_val == ""):
            cut_val = 3600
        else:
            if(isfloat(cut_val)):
                cut_val = isfloat(cut_val)
            else:
                raise ValueError('Invalid cutting distance')
                
        output_folder = self.output_line_edit.text()
        if (output_folder == ""):
            output_folder = os.path.join(os.getcwd(), "output_folder")
            if (os.path.isdir(output_folder)):
                pass
            else:
                os.mkdir(output_folder)
        else:
            output_folder = os.path.join(os.getcwd(), output_folder)
            if (os.path.isdir(output_folder)):
                pass
            else:
                os.mkdir(output_folder)
        
        file = open("inputarguments", "w")
            
        file.write(input_file+"\n")
        file.write(str(Synthetic_run)+"\n")
        file.write(str(synthetic_error) + "\n")
        file.write(str(real_data_run)+"\n")
        file.write(str(kink_val)+"\n")
        file.write(str(cut_val)+"\n")
        file.write(str(N_subspace)+"\n")
        file.write(output_folder)
        file.close()
        script = os.path.join(os.getcwd(), "main_modified_PSO.m")

        # Set the command to run the script
        command = [matlab_run, "-nosplash", "-nodesktop", "-wait", "-r", f"run('{script}');exit;"]

        # Run the command and wait for it to finish
        subprocess.run(command)

        # Run the Python script
        pyscript = os.path.join(os.getcwd(), "pso_art_fig_out.py")
        print("Runing Python script for ploting figures. It can take some time be patient")
        command = [sys.executable, pyscript]
        subprocess.run(command)
        self.close_signal.emit()

        
        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyUI()
    sys.exit(app.exec_())