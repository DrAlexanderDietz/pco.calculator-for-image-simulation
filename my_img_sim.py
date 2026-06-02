import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from PIL import Image
import tifffile as tiff
import csv, os
import csv
from matplotlib import rcParams
import math
import io

# -----------------------------
# Utility functions 
# -----------------------------

st.set_page_config(
    page_title="pco.calculator Image Simulation",
    page_icon="Resources/Flash_comp.png"   # Path to a local .png, .jpg, or .ico file
)

def make_sidebar():
    """Calling this function draws the sidebar with all its settings and parameters and returns their values as dict."""

    def load_csv_to_dict(file_path):
        """
        Import data from csv file from <file path>
            
        Parameter:
        file_path: path/name of import csv file
            
        """
        
        data_dict = {}
        
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile,delimiter=';')
            next(reader)  # Skip header row
            
            #extract from csv. Use the camera+mode as key
            for row in reader:
                if row:
                    key = row[1]
                    values = [row[0]]
                    values += [float(i) for i in row[2:]]
                    data_dict[key] = values
        
        return data_dict 

    def load_QE_curve(file_path, model):        
        """
        Read the QE[%]=f(wavelength) from txt file. Use the camera family 
        'model'as the basis for the curve choice.
        
        Parameter:
            file_path: path/name of import csv file
            model: camera model such as 'pco.edge 4.2'
        """
        
        datei = open(file_path + "/" + model, "r")
        
        my_dict = {}         
        zeilen = datei.readlines()             
        
        for i in range(2):
            zeilen.pop(0)                      
        
        for zeile in zeilen:
            werte = zeile.split()
            my_dict[float(werte[0].replace(',','.'))] = float(
                werte[-1].replace(',','.'))
            
        datei.close()
            
        return(my_dict)
                    
            
    csv_file = 'Resources/CamsList.csv'  # Replace with your CSV file path
    data_cams = load_csv_to_dict(csv_file) #load camera datasheet values from file

    def data_sheet_vals(cam):
        
        """
        This function shall update return specification variables/items when a camera model 
        is chosen in the GUI. The values are determined by the camera datasheet.

        Ouput: list[pixel pitch, full well cap, read noise, dark noise,conversion factor, offset, dsnu, ADC]
        
        """
        
        #simple overwriting of pvalues according to dataset value
        
        ds_pxl = float(data_cams[cam][1])
        ds_fwc = int(data_cams[cam][2])
        ds_ron = float(data_cams[cam][3])
        ds_mu_d = float(data_cams[cam][4])
        ds_cF = float(data_cams[cam][6])
        ds_dno = int(data_cams[cam][7])
        ds_dsnu = float(data_cams[cam][8])
        ds_ADC = int(data_cams[cam][9])

        return([ds_pxl,ds_fwc,ds_ron,ds_mu_d,ds_cF,ds_dno,ds_dsnu,ds_ADC])

    def get_qe(cam, wavelength):
        """
        Pull QE-Curve Data from raw files and return as function of camera class and wavelength.
        Interpolates between nearest wavelength points.
        
        Parameters:
        - cam: camera model
        - wavelength: wavelength in nm
        """

        target = float(wavelength)
        camera_class = data_cams[cam][0]

        # Import from txt file
        qe_dict = load_QE_curve("Resources", f"{camera_class}.txt")
        keys = sorted(qe_dict.keys())

        # Clamp if outside the known range
        if target <= keys[0]:
            qe_cam_wl = qe_dict[keys[0]]
        elif target >= keys[-1]:
            qe_cam_wl = qe_dict[keys[-1]]
        else:
            # Find neighbors and interpolate
            for i in range(len(keys) - 1):
                k1, k2 = keys[i], keys[i + 1]
                if k1 <= target <= k2:
                    v1, v2 = qe_dict[k1], qe_dict[k2]
                    qe_cam_wl = v1 + (v2 - v1) * (target - k1) / (k2 - k1)
                    break

        return round(qe_cam_wl / 100, 2)  # convert from % → fraction
        
    def crop_min(exp_n_in):
        
        "callback function for setting crop to reasonable values only --> clamp min based on the width of the image as 2^n"
        
        clamp_min = math.ceil(min(((2**(exp_n_in-10))*100), 100))
        
        return clamp_min
    
    #### Make the sidebar ########-----------------------
    
    # Sidebar title
    st.sidebar.title("Controls")

    # ---------- IMAGE SETTINGS ----------
    st.sidebar.subheader("IMAGE SETTINGS")

    # Dropdown for image selection
    dd_img_options = [
        "Gaussian", "Square", "Homogeneous", "Test Pattern HDR", "Microscopy Example",
        "Astronomy Example", "Testchart Example", "Upload Image"]
    dd_img = st.sidebar.selectbox("Choose Image",
                                dd_img_options,
                                help="Example image options may be based on 8-bit image data!")

    if dd_img == "Upload Image":
        upl_file_item = st.sidebar.file_uploader(
        "Upload any Image - Please see info",
        type=["png", "jpg", "jpeg"],
        key="uploaded_file",
        help="Please use only images that are larger than 1064 x 1064 pxl!"
        )

    # --- Slider with callbacks ---
    st.sidebar.slider("$log_2$(Image width)",
            5, 10, 8,
            key="exp_n",
            help="Set quadratic ROI (with 2$^n$ pixel) with possible width values of 32/ 64/ 128/ 256/ 512/ 1064.")

    #disablbe slider when full 1064 pxl width image is schosen
    if st.session_state.exp_n != 10:
        st.sidebar.slider("Pixel Size Coefficient [%]",
            crop_min(st.session_state.exp_n), 100, 100,
            key="crop",
            help="Adjusting the **pixel size coefficient** allows to compare different pixel sizes. If you"\
            " want to compare 4.6 µm pixels of pco.edge 10 bi with the 6.5 µm pixels of" \
            " pco.edge 4.2 for example, simply apply a 71% 'coefficient' (=4.6/6.5) to the larger pixels to adapt the field" \
            " of view and effectively 'zoom out' in the image. The initial FoV is also shown as green box."
            )
    else:
        st.sidebar.slider("Pixel Size Coefficient [%]",
            1, 100, 100,
            key="crop",
            disabled=True
            )

    # Line profile position
    slider_linpos = st.sidebar.slider("Line Profile X [Height %]", 0, 99, 50, 
                                      help="Position of the line profile relative to the image height")
    
        # Line profile position
    slider_linpos_II = st.sidebar.slider("Line Profile Y [Width %]", 0, 99, 50, 
                                      help="Position of the line profile relative to the image width")

    # ---------- CAMERA & EXPERIMENT ----------
    st.sidebar.subheader("CAMERA & SETTINGS")

    # Camera dropdown
    dropdown_options = list(data_cams.keys())
    camera_model = st.sidebar.selectbox("Camera Model and Mode",
                                        dropdown_options,
                                        index=len(dropdown_options)-1,
                                        help="Chose your camera model. For all PCO cameras / opertaion modes values are predefined."\
                                            " To configure your own custom camera use the **sCMOS** option. The Perfect Camera does" \
                                            " act for comparison purposes. It does not show any tecnical noise besides the Poissonian" \
                                            " noise and has perfect QE.")

    # Exposure time
    exposure_time = st.sidebar.text_input("Exposure Time / sec", "1.0",
                                          help = "Set exposure time according to your experiment. Together with the maximum photon flux"
                                          "density and background illumination this will determine the amount of photons that will" \
                                          "arrive at the dtector. The amount of dark current building up during" \
                                          "exposure is also a consequence of exposure time.")

    # Binning options
    bin_values_list = ["1x1", "2x2", "4x4"]
    bin_opts = st.sidebar.selectbox("Binning", bin_values_list, 
                                    help = "In sCMOS sensors, binning is only applied after acquisition and thus only " \
                                        "provides a statistical improvement of the signal-to-noise ratio. Binning of four pixels "
                                        "(2x2) will increase the SNR by a foactor of 2!")

    # Averaging
    avg_numb = st.sidebar.number_input("Number Images for Averaging", 1, 
                                       help = "Averaging over series of images will reduce the total noise by a factor of one over the square " \
                                       "root of the number of images. For example averaging over 16 images will increase the signal-to-noise ratio" \
                                       " by a factor of 4!")   

    # ---------- ILLUMINATION ----------
    st.sidebar.subheader("ILLUMINATION")

    #wavelength as slider
    wavelength = st.sidebar.slider("Wavelength / nm", 200, 1100, 600,
                                   help = "For the simulation we assume illumination to be monochromatic. The respective quantum efficiency of the PCO" \
                                   " camera of choice will be retreived automatically.")

    #photon flux density max and add backgraund
    phi_pfd_max = st.sidebar.text_input("Max. Photon Flux Density / ph/(um)²/sec", "1.0",
                                        help = "The photon flux density describes the number of photons that arrive at an area of 1 µm x 1 µm " \
                                        "per second. This definition helps to properly compare the overall sensitivities of cameras with different pixel" \
                                        " densities. Adjust the **maximum photon flux density** to control the maximum illumination in the ground truth image.")

    phi_pfd_bg = st.sidebar.text_input("Background Illumination / ph/(um)²/sec", "0.0", 
                                       help = "The **background illumination** control allows to superimpose the input ground truth illumination with an"
                                       "additional uniform background-light contribution. This helps to deliberately introduce or counterbalance any offset " \
                                       "in the input image." )

    # ---------- DISPLAY SETTINGS ----------
    st.sidebar.subheader("DISPLAY SETTINGS")

    with st.sidebar:
        with st.expander("Plot & LUT Details"):
            #histogram scale choices Linear or Log
            hist_scale = st.selectbox("Histogram Scale", ["Linear", "Logscale"])

            #Autoscale the image data or set deliberate limits 
            ats_opts_list = ["Autoscale","Set LUT Limits"]
            ats_opts = st.selectbox("Scaling", ats_opts_list)

            #disable when autoscale
            if ats_opts == "Autoscale":
                disable_lut_widget = True
            else:
                disable_lut_widget = False 

            #max and min for manual LUT
            lut_max = st.text_input("Scale LUT max.", "255", disabled=disable_lut_widget)
            lut_min = st.text_input("Scale LUT min.", "0", disabled=disable_lut_widget)

    # ---------- CAM SPECIFICATIONS ----------
    st.sidebar.subheader("CAMERA SPECIFICS")

    with st.sidebar:
        with st.expander("Camera Details"):

            #disable als camera spex settings unless choice is sCMOS
            if camera_model != "sCMOS":
                disable_widget = True
            else:
                disable_widget = False

            #make camera spex as sidbar items
            qe = st.slider("Quantum Efficiency", min_value=0.00, max_value=1.00, value=float(get_qe(camera_model, wavelength)),
            step=0.01,disabled=disable_widget)
            
            pxlpitch = st.text_input("Pixel Pitch / µm", data_sheet_vals(camera_model)[0], disabled=disable_widget,
                                     help = "The **pixel pitch** represents length and height of the individual pixels (detector elements) of"
                                     " the image sensor.")
            

            rn = st.text_input("Read Noise / e-/pxl", data_sheet_vals(camera_model)[2], disabled=disable_widget,
                                    help = "Read Noise is an inherent feature of CMOS image sensors. In this simulation the distribution of the read noise" \
                                    " is approximated as a normal distribution. In reality, read noise may deviate from the perfect Gauss curve to some degree," \
                                    " depending on the type of sensor utilized.")

            dc = st.text_input("Dark Current / e-/pxl/sec", data_sheet_vals(camera_model)[3], disabled=disable_widget,
                                    help = "The Dark Current describes the number of thermal electrons created in a pixel per second exposure. As a rule-" \
                                    "of-thumb this value doubles every 7-8 °C when increasing the sensor temperature. Dark Current shows a Poissonian " \
                                    "distribution and therefore contributes to the total noise of the camera.")
            
            fwc = st.text_input("Fullwell Capacity / e-", data_sheet_vals(camera_model)[1], disabled=disable_widget,
                                        help = "Fullwell Capacity in units of electrons. If the signal is clipping, the respective pixels will be highlighted" \
                                        " in red in the simulated image!")
            
            bit_depth = st.slider("ADC Bit-Depth", min_value=8,max_value=16, value=int(data_sheet_vals(camera_model)[7]),
                                        step=1, disabled=disable_widget, 
                                        help = "Dynamic Range of the analog-to-digital coverter. Image data will clip at this level.")
            
            convF = st.text_input("Conversion Factor / e-/DN", data_sheet_vals(camera_model)[4], disabled=disable_widget,
                                        help = 'The Conversion Factor is the inverse analog of the also often used "System Gain". It describes how many'
                                        ' photo-electrons are required to increase the signal by one gray level.')
            
            dn_offset = st.text_input("DN Offset", data_sheet_vals(camera_model)[5], disabled=disable_widget,
                                            help = "To prevent signal from clipping at the dark end, an artificial offset is set to shift the histogram to the " \
                                            " right by a certin number of gray levels. This dark offset would need to be subtracted for any kind of ratio-" \
                                            "metric analyses.")
            
            drksnu = st.text_input("DSNU / e-", data_sheet_vals(camera_model)[6], disabled=disable_widget,
                                        help="Total Dark Signal Non-Uniformity. For simplicity we assume only random pixel and columnwise fixed pattern!")
            
            dsnu_cont_sldr = st.slider("DSNU Pixel : Column Ratio [%]",min_value=0, max_value=100, value=0,
                                            help = 'This slider adjusts the the ratio for the DSNU origin sources: 100 percent means randomly distributed non-uniformities,' \
                                            ' whereas 0 percent refers to a purely columnwise fixed non-uniformity pattern.')
            


    # ---------- SIMULATION CONTROL ----------
    st.sidebar.subheader("SIMULATION DOWNLOAD")

    #choice for download options
    save_values_list = ["Simulated Image as TIFF", "Simulation Summary PDF"]
    export_option = st.sidebar.selectbox("Image Export Format", save_values_list)

    #is SCMOS is choice use slider/input setting, otherwise pull datasheet values
    if camera_model == "sCMOS":
        qe_eff =  float(qe)
        ron = float(rn)
        mu_dark =  float(dc)
        full_well_cap = float(fwc)
        cF =  float(convF)
        p_pitch =  float(pxlpitch)
        dno =  float(dn_offset)
        dsnu = float(drksnu)
        dsnu_cs = float(dsnu_cont_sldr)
        adc = int(bit_depth)
    else:
        qe_eff =  get_qe(camera_model, wavelength)
        ron = data_sheet_vals(camera_model)[2]
        mu_dark =  data_sheet_vals(camera_model)[3]
        full_well_cap = data_sheet_vals(camera_model)[1]
        cF =  data_sheet_vals(camera_model)[4]
        p_pitch =  data_sheet_vals(camera_model)[0]
        dno =  data_sheet_vals(camera_model)[5]
        dsnu = data_sheet_vals(camera_model)[6]
        dsnu_cs = float(dsnu_cont_sldr)
        adc = data_sheet_vals(camera_model)[7]

    return {
        # image
        "base_image": dd_img,
        "exp_n": int(st.session_state.exp_n),
        "f_width": int(2 ** st.session_state.exp_n),
        "img_comp": int(st.session_state.crop)/100,
        "line_pos_prct": int(slider_linpos),
        "line_pos_II_prct": int(slider_linpos_II),
        # camera & experiment
        "camera_name": camera_model,
        "t_exp": float(exposure_time),
        "bin_factor": bin_values_list.index(bin_opts),
        "avg_img_numb" : int(avg_numb),
        # illumination
        "lmda_nm": int(wavelength),
        "phi_pfd_max": float(phi_pfd_max),
        "phi_pfd_bg": float(phi_pfd_bg),
        # display
        "hist_scale": hist_scale,
        "lut_autoscale": ats_opts_list.index(ats_opts),
        "lut_scale_max": float(lut_max),
        "lut_scale_min": float(lut_min),
        # cam specs
        "qe_eff": qe_eff,
        "ron": ron,
        "mu_dark": mu_dark,
        "full_well_cap": int(full_well_cap),
        "convF": cF,
        "pxl_pitch": p_pitch,
        "dn_offset": int(dno),
        "drksnu": dsnu,
        "dsnu_cont_sldr" : dsnu_cs,
        "adc_bit" : int(adc),
        "export_choice": export_option,
    }

def bin_fac(input_vals):
    """#Bining als Potenz von 2: 0: 1x1 / 1: 2x2 / 2: 4x4, ..."""
    return(2**input_vals["bin_factor"])

def eff_el_ph(input_vals):
    """#Max Mean Photons from Imgage: eff_el_ph = phi_pfd_max * t_exp * pxl_pitch**2 """
    
    phi_pfd_max = input_vals["phi_pfd_max"]
    t_exp = input_vals["t_exp"]
    pxl_pitch = input_vals["pxl_pitch"]
    
    return(phi_pfd_max * t_exp * pxl_pitch**2)

def eff_el_bg(input_vals):
    """Mean Photon Contribution from homogeneous Background: eff_el_bg = phi_pfd_bg * t_exp * pxl_pitch**2"""
    
    phi_pfd_bg = input_vals["phi_pfd_bg"]
    t_exp = input_vals["t_exp"]
    pxl_pitch = input_vals["pxl_pitch"]
    
    return(phi_pfd_bg * t_exp * pxl_pitch**2)

def line_pos(input_vals):
    """#Effektive Position [% der Bildhöhe] der Linie auf Canvas"""
    
    f_width = input_vals["f_width"]
    lpp = input_vals["line_pos_prct"]
    
    return(int(f_width/bin_fac(input_vals)*lpp/100))

def line_pos_II(input_vals):
    """#Effektive Position [% der Bildhöhe] der Linie auf Canvas"""
    
    f_width = input_vals["f_width"]
    lpp = input_vals["line_pos_II_prct"]
    
    return(int(f_width/bin_fac(input_vals)*lpp/100))

def lut_settings(input_vals):
    """Conacntenate the LUT settings to list:
        [ON/OFF, MAX LUT value, MIN LUT value]"""
    
    return([input_vals["lut_autoscale"],input_vals["lut_scale_max"],
            input_vals["lut_scale_min"]])

def rand_pG(mu, input_vals):     
        """
        calculate random pixel read value based on mean mu, conversion factor 
        convF = 1/K, readout noise / e-, dark current mu_dark
        """       
        convF = input_vals["convF"]
        ron = input_vals["ron"]
        mu_d =input_vals["mu_dark"]
        t_e = input_vals["t_exp"]
        dn_offset = input_vals["dn_offset"]
        avg_img_numb = input_vals["avg_img_numb"]
                       
        avg_random_no = 0 #container for averaging

        for i in range(avg_img_numb): #loop over number of images
            
            #discrete poisson distrubuted value
            poisson_rand_no = np.random.poisson(mu+t_e*mu_d)

            #gauss curve around discrete poisson value with read noise as stdv w/ discrete DN
            gauss_smeared_no = np.round(np.random.normal(poisson_rand_no,ron)*1/convF)+dn_offset
        
            #sum up all random values        
            avg_random_no = avg_random_no + gauss_smeared_no
        
        #finally divide by numer of imgs.
        avg_random_no = np.round(avg_random_no/avg_img_numb)
        
        return(avg_random_no)
    
def gaussian_array(x, y, sigma, hgt=1):
    """
    Generates a 2D array of shape (x, y) following a Gaussian distribution.
    The maximum value is close to the center of the image.
    
    Parameters:
        x (int): Number of rows in the array.
        y (int): Number of columns in the array.
        sigma (float): Standard deviation of the Gaussian distribution.
    
    Returns:
        np.ndarray: 2D array with Gaussian distribution.
    """
    xv, yv = np.meshgrid(np.arange(y), np.arange(x))
    x0, y0 = x / 2.1, y / 2.1
    gaussian = hgt * np.exp(-((xv - x0) ** 2 + (yv - y0) ** 2) / (2 * sigma ** 2))

    return gaussian    

def square_overlay(input_vals, dw=0.5, hgt=1):
    """Creates a map for a brightsquare shaped area"""
    
    width = input_vals["f_width"]
    crop = input_vals["img_comp"]
    
    sq_image = np.zeros((width,width))
    
    #Indizes für das Plateau berechnen       
    start = int(((width - int(width*dw*crop)) // 2))
    end = int((start + int(width*dw*crop)))

    # Plateau mit Einsen füllen
    sq_image[start:end, start:end] = hgt
    
    return sq_image
    
def homogeneous_illumination(input_vals):
    """Creates a map for a homogeneous illumination"""
    
    width = input_vals["f_width"]
    
    sq_image = np.ones((width,width))
    
    return sq_image

def test_chart_artificial(input_vals):
    """
    Generates an array that spans 15-bit of dynamic range.
    """
    size = input_vals["f_width"]
    sectors = 64

    # Create coordinate grid
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Convert to polar coordinates
    theta = np.arctan2(Y, X)
    
    # Create radial sector pattern (black and white alternating sectors)
    pattern = (np.sin(sectors * theta) > 0).astype(int)
    
    pattern = (pattern+7)/8 #make pattern alternate 1/8
    
    i, j = np.indices(pattern.shape)
    diag_mask = i < j
    
    pattern = pattern * diag_mask + diag_mask.T # show pattern only on one diagonal
    
    frame_blank = np.zeros((size,size//16))       
    frame_karo = np.concatenate((frame_blank,frame_blank+2**1,frame_blank+2**2,frame_blank+2**3,
                                 frame_blank+2**4,frame_blank+2**5,frame_blank+2**6,frame_blank+2**7,frame_blank+2**8,
                                 frame_blank+2**9,frame_blank+2**10,frame_blank+2**11,frame_blank+2**12,frame_blank+2**13,
                                 frame_blank+2**14, frame_blank+2**15),axis=1)  
    
    frame_karo += frame_karo.T

    frame_total = frame_karo * pattern

    frame_total = frame_total / frame_total.max()
    
    return frame_total

def any_image(image_path, input_vals):
    """"Allows to load any image and perform the the noise overlay"""
    
    width = input_vals["f_width"]
    comp_fac = input_vals["img_comp"]
    
    # Open the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale for (x, y) dimensions
    
    # Define the compression fraction (e.g., 0.5 = reduce to 50% of original size)
    fraction = comp_fac
    
    # Get original size
    original_width, original_height = img.size
    
    # Calculate new size
    new_width = int(original_width * fraction)
    new_height = int(original_height * fraction)
    
    # Resize image
    comp_img = img.resize((new_width, new_height))
    
    comp_img = np.array(comp_img)
    comp_img = comp_img/comp_img.max() #Scale to max val = 1
    
    #centercrop image
    y,x = comp_img.shape
    startx = x//2 - width//2
    starty = y//2 - width//2  
    comp_img = comp_img[starty:starty+width, startx:startx+width]
    
    return comp_img

def bin_array_sum(array,bin_size):
        """
        Bins a quadratic numpy array by summing over non-overlapping bins.
        
        Parameters:
            array (np.ndarray): Input 2D quadratic array.
        
        Returns:
            np.ndarray: Binned array with summed values.
        """
        
        x, y = array.shape
        if x % bin_size != 0 or y % bin_size != 0:
            raise ValueError("Array dimensions must be divisible by bin size.")
        
        reshaped = array.reshape(x // bin_size, bin_size, y // bin_size, bin_size)
        return reshaped.sum(axis=(1, 3))

def make_plots(new_vals):
    """This is the main function to run the plot generation. It draws its
    values from the values set in the GUI."""
    
    #extract variables
    qe_eff = new_vals["qe_eff"]
    full_well_cap = new_vals["full_well_cap"]
    convF = new_vals["convF"]
    export_choice = new_vals["export_choice"]
    ron = new_vals["ron"]
    mu_dark = new_vals["mu_dark"]
    t_exp = new_vals["t_exp"]
    dn_offset = new_vals["dn_offset"]
    drksnu = new_vals["drksnu"]
    camera_name = new_vals["camera_name"]
    img_comp = new_vals["img_comp"]
    hist_scale = new_vals["hist_scale"]
    lmda_nm = new_vals["lmda_nm"]
    pxl_pitch = new_vals["pxl_pitch"]
    avg_numb = new_vals["avg_img_numb"]
    
    #plot "truth" left and simulated image right
    fig, axs = plt.subplots(2, 2,
                            figsize=(14,11),  #Groesse Abb
                            gridspec_kw={'width_ratios': [1,1], 
                                            'height_ratios': [2,1]}, #Groessenverhaeltnis
                            )
    
    #Provide all hidden info on canvas as suptitle
    fig.suptitle(t="Simulation {}".format(camera_name)+"\n"
                    +"Exposure: {} s |".format(t_exp)
                    +" Pixel Size Coefficient: {} | Average over {} Images ".format(img_comp,avg_numb) + "\n"
                    +"Read Noise: {} e- |".format(ron)
                    +" Dark Current: {} e-/sec |".format(mu_dark)                   
                    +" QE: {}% @ {}nm |".format(round(qe_eff*100),lmda_nm)
                    +" Conversion: {} e-/DN |".format(convF)
                    +" DNSU: {} e- |".format(drksnu)
                    +" Offset: {} DN |".format(round(dn_offset))
                    +" Pixel Size: {} µm ".format(pxl_pitch),
                    #+"\n",
                    fontsize='small',
                    ha='left',
                    y=0.98,
                    x=0.02,
                    font='monospace'
                    )    

    rcParams['font.weight'] = 'light'
    rcParams["figure.autolayout"] = True
    plt.style.use("classic")
    
    def crop_window(fw, crop, plt_no, binf=0):

        """Visualizes a green square on the image  to indicate the original FoV w/o any image 
        compression via pixel size scaling.
        
        Parameters:
            fw: frame width
            crop: pixel size coefficient
            plt_no: index of plot
            binf: binning factor
        """
        
        if crop != 1:
            crf_max = fw*(crop+(1-crop)/2)/(2**binf)
            crf_min = fw*(1-crop)/(2**binf)/2
            axs[plt_no].plot([crf_min,crf_max,crf_max,crf_min,crf_min],
                            [crf_min,crf_min,crf_max,crf_max,crf_min],
                            '-', color='limegreen', alpha=1)

    def truth_plot(phots, input_vals, plt_no=(0,0),
                    pos=line_pos(new_vals), posII=line_pos_II(new_vals), bf=bin_fac(new_vals)):       
        """
        Plots the Photon ground truth 2D distribution as a color map.
        
        Parameters:
            phots: photon map as input quadratic 2D array
            plt_no: index of plot on subplots
            new_vals: camera parameter dict as input
            pos: position of line profile indicator
            bf: binning factor required for line profile positioning                 
        """
        
        fw = input_vals["f_width"]
        crop = input_vals["img_comp"]

        phots = axs[plt_no].imshow(phots,
                                cmap='rainbow',
                                interpolation='none')
        axs[plt_no].plot([0,fw],[pos*bf,pos*bf],'-', color='gold')
        axs[plt_no].plot([posII*bf,posII*bf],[0,fw],'m-')
        axs[plt_no].autoscale(False)
        axs[plt_no].xaxis.tick_top()
        axs[plt_no].set_xlabel("Input: Expected Photons per Phys. Pixel", labelpad=10)
        axs[plt_no].minorticks_on()
        
        crop_window(fw, crop, (0,0))
            
        #axs[plt_no].set_xlabel(r"Ground Truth", labelpad=10)
        fig.colorbar(phots, fraction=0.046)
    

    def generate_sim_im_plot(img,input_vals, plt_no=(0,1)):
        
        """
        Plots the simulate image in DN as a grayscale map with overlay for
        blown out pixels.
        
        Parameters:
            img: DN map as input quadratic 2D array
            plt_no: index of plot on subplots
        """
        
        fwc= input_vals["full_well_cap"]
        cF= input_vals["convF"]
        fw= input_vals["f_width"]
        crop = input_vals["img_comp"]
        bin_fac = input_vals["bin_factor"]
        bit_depth = input_vals["adc_bit"]
        dno = input_vals["dn_offset"]
        #cam_name= input_vals["camera_name"]
        
        pos=line_pos(new_vals)
        posII =line_pos_II(new_vals)
        luts = lut_settings(new_vals)                         
        
        #Apply LUT limits if wanted
        if luts[0] == 0:
            v_max_img, v_min_img = img.max(),img.min()
        else:
            v_max_img, v_min_img = luts[1],luts[2]
        
        
        #Identify every pixel that exceeds full well capacity / bit depth...
        mask = img > fwc*1/cF
        #mask = img >= 2**bit_depth-dno
        overlay = np.zeros((*img.shape, 4))
        overlay[mask] = [1,0,0,1]
                      
        #Gray Scale Image scaled according to LUT limits
        img = axs[plt_no].imshow(frame_img,
                            cmap = 'gray',
                            interpolation='none',
                            vmin=v_min_img,
                            vmax=v_max_img,
                            )
        
        #...and overlay in red
        axs[plt_no].imshow(overlay,interpolation='none')
        axs[plt_no].imshow(frame_img,alpha=0,interpolation='none',)
        axs[plt_no].plot([0,fw],[pos,pos],'r-')
        axs[plt_no].plot([posII,posII],[0,fw],'b-')

        #Plot Settings
        axs[plt_no].xaxis.tick_top()
        axs[plt_no].autoscale(False)
        axs[plt_no].set_xlabel("Simulated Image / DN", labelpad=10)
        #axs[plt_no].set_ylabel(r"{}".format(cam_name), labelpad=10)
        fig.colorbar(img, fraction=0.046) 
        
        crop_window(fw,crop, (0,1), bin_fac)                               

    def generate_line_profile_plot(img,phots,input_vals, plt_no=(1,0)):
        """
        Plots the line profiles for truth (if converted to DN) and simulated image
        
        Parameters:
            img: DN map as input quadratic 2D array
            phots: photon map as input quadratic 2D array
            plt_no: index of plot on subplots
            pos: position of line profile indicator
            bf: binning factor required for line profile positioning  
            cF: conversion factor in e-/DN
            md: dark current in e-/secs
            te: exposure time in secs
            dno: dark offset in DN to ensure no clipping at zero happens due to noise               
        """
        
        pos = line_pos(new_vals)
        posII = line_pos_II(new_vals)
        cF = input_vals["convF"]
        bf = bin_fac(new_vals)
        md = input_vals["mu_dark"]
        te = input_vals["t_exp"]
        dno = input_vals["dn_offset"]       

        luts = lut_settings(new_vals)                         

        #Apply LUT limits if wanted
        if luts[0] == 0:
            v_max_img, v_min_img = img.max(),img.min()
        else:
            v_max_img, v_min_img = luts[1],luts[2]
            
        #Line prof values (y) for simulated and truth image
        line_prof_vals = img[pos].tolist()
        line_prof_vals_truth = ((1/cF*(phots[pos*bf]*qe_eff+md*te)+dno)*bf**2).tolist()
        
        line_profII_vals = img[:, posII].tolist()
        line_profII_vals_truth = ((1/cF*(phots[:, posII*bf]*qe_eff+md*te)+dno)*bf**2).tolist()
        
        #Line prof values (x) for simulated and truth image
        row_no = [i for i in range(len(line_prof_vals))]
        row_no_truth = [i/bin_fac(new_vals) for i in range(len(line_prof_vals_truth))]
        
        #Plot Settings
        axs[plt_no].plot(row_no,line_prof_vals,'r-', label='Simulated Image (X)')
        axs[plt_no].plot(row_no,line_profII_vals,'b-', label='Simulated Image (Y)')
        axs[plt_no].plot(row_no_truth,line_prof_vals_truth,'-',color='gold', label='Ground Truth (X)')
        axs[plt_no].plot(row_no_truth,line_profII_vals_truth,'-',color='magenta', label='Ground Truth (Y)')
        axs[plt_no].set_xlim([0,row_no[-1]])
        axs[plt_no].set_ylim([v_min_img,v_max_img]) #new
        axs[plt_no].yaxis.tick_left()
        axs[plt_no].xaxis.tick_bottom()
        axs[plt_no].set_xlabel(r"Line Profile Image / DN", labelpad=10)
        axs[plt_no].xaxis.set_label_position('bottom')
        axs[plt_no].set_ylabel(r"", labelpad=30)
        axs[plt_no].legend(fontsize=8,frameon=False, loc="upper left")
        
    
    def generate_histogram_plot(img,plt_no=(1,1), save_img=export_choice,
                                scale=hist_scale, new_vals=new_vals):
        """Plot histogramm mit ganzzahligen bins, da Wertebereich als DN bzw. Ganzzahl
        
        Parameters:
            img: DN map as input quadratic 2D array
            plt_no: index of plot position on canvas
            scale: chose the scale of the y-axis 'Linear' or 'Logscale'
        """
        
        luts = lut_settings(new_vals)                         

        #Apply LUT limits if wanted
        if luts[0] == 0:
            v_max_img, v_min_img = img.max(),img.min()

        #prevent error if only one bin is filled in histogramm
            if (v_max_img - v_min_img) < 16:
                v_max_img = v_max_img + 4
                v_min_img = v_min_img - 4                           
                                    
        else:
            v_max_img, v_min_img = luts[1],luts[2]

        #Make a list for the histogram y-vals with all pixel values
        y_vals=[]
        
        for i in range(img.shape[1]):
            for j in range(img.shape[1]):
                y_vals.append(int(frame_img[i][j]))
        
        #Make a destinction for the histogramm settings to ensure a nice look and limit data x-axis data points to 500       
        if v_max_img - v_min_img > 500:
            bin_no = 'auto'
        else:
            bin_no = [i for i in range(int(v_min_img),int(v_max_img),1)]

        #plot settings    
        my_hist, bins, patches = axs[plt_no].hist(y_vals,
                            log=(scale=="Logscale"),
                    bins = bin_no,
                    density="True",
                    color="steelblue",
                    linewidth=0.0,
                    histtype='stepfilled',
                    )
        max_val= my_hist.max()

        axs[plt_no].plot(np.linspace(v_min_img,v_max_img,100),np.linspace(0,max_val,100),'-',color='black', alpha=0.5)
        axs[plt_no].yaxis.tick_left()
        axs[plt_no].xaxis.tick_bottom()
        axs[plt_no].xaxis.set_label_position('bottom')
        #axs[plt_no].set_ylabel(r"Frequency", labelpad=0)
        axs[plt_no].set_xlabel(r"Histogram / DN", labelpad=10)
        axs[plt_no].autoscale(enable=True, axis='x', tight=False)
        axs[plt_no].locator_params(axis='x', nbins=6)
        axs[plt_no].set_xlim(v_min_img,v_max_img)
        
        plt.tight_layout()
        
        plt.text(0.03, 0.97, "Mean: "+str(round(np.mean(y_vals),1)),
                    transform=plt.gca().transAxes,
                    fontsize=12, verticalalignment='top',
                    horizontalalignment='left')

        plt.text(0.03, 0.9, "StDv: "+str(round(np.std(y_vals),1)),
                    transform=plt.gca().transAxes,
                    fontsize=12, verticalalignment='top',
                    horizontalalignment='left')
        
        #plt.show()

        #draw the downloadbutton and award it functionality to download the pdf to the above plots
        if save_img == "Simulation Summary PDF":
            
            #plt.savefig('Export_image.png', dpi=300)

            buf = io.BytesIO()
            fig.savefig(buf, format="pdf")
            buf.seek(0)


            st.sidebar.download_button(
            "DOWNLOAD SUMMARY PDF",
            data=buf,
            file_name="Image_Simulation.pdf",
            mime="application/pdf",
            width='stretch',
                )
                       

    def get_base_image(input_vals):
            """
            Draw the image that shall be superimposed with noise from a 
            selection.
            
            Parameter:
                - input vals: import all important camera spex as dict
                - path: data path for choice and grabbing data from an image file
            """
                        
            base_img = input_vals["base_image"]
            f_width = input_vals["f_width"]
            img_comp = input_vals["img_comp"]
            
            if base_img == "Gaussian":
                return(gaussian_array(f_width,f_width,f_width/5*img_comp) * eff_el_ph(new_vals) + eff_el_bg(new_vals))
            
            elif base_img == "Homogeneous":
                return(homogeneous_illumination(input_vals) * eff_el_ph(new_vals) + eff_el_bg(new_vals))
            
            elif base_img == "Square":
                return(square_overlay(input_vals) * eff_el_ph(new_vals) + eff_el_bg(new_vals))
            
            elif base_img == "Test Pattern HDR":
                return(test_chart_artificial(input_vals) * eff_el_ph(new_vals) + eff_el_bg(new_vals))
            
            elif base_img == "Microscopy Example":
                return(any_image('import_images/image_bio.png',input_vals) * eff_el_ph(new_vals) + eff_el_bg(new_vals))
            
            elif base_img == "Astronomy Example":
                return(any_image('import_images/image_space.jpg',input_vals) * eff_el_ph(new_vals) + eff_el_bg(new_vals))
            
            elif base_img == "Testchart Example":
                return(any_image('import_images/test-chart-eia.jpg',input_vals) * eff_el_ph(new_vals) + eff_el_bg(new_vals))
            
            elif base_img == "Upload Image":
                return(any_image(st.session_state.uploaded_file, input_vals) * eff_el_ph(new_vals) + eff_el_bg(new_vals))
               
    def safe_as_tiff(sim_im):
            """
            This function simply saves the outputimage as a TIFF that can
            later be opened in for example FIJI for an in depth comparison.
            """
            
            im_export = Image.fromarray(sim_im)

            buf = io.BytesIO()
            #im_export.save(buf, format="TIFF")
            tiff.imwrite(buf, sim_im.astype(np.uint16))
            buf.seek(0)

            # Create download button
            st.sidebar.download_button(
            label="DOWNLOAD TIFF",
            data=buf,
            file_name="pco_simulated_image.tif",
            mime="image/tiff",
            width='stretch',
            )  
         

    #Start with generating the arrays for the base and noisy image
    frame_phots = get_base_image(new_vals)

    #create an array with a fixed pattern for simulation of DSNU
    DSNU_var = new_vals["drksnu"]

    #Column located DSNU
    pattern_noise_list = np.loadtxt("Resources/normal_random_1024.txt", delimiter=",") #base file: 1D random distribution w/ stdv 1 around 0
    pattern_noise_list = pattern_noise_list[0:frame_phots.shape[-1]] #truncate to base image width
    pattern_noise_frame = np.array(pattern_noise_list)[None, :] * np.ones((frame_phots.shape[-1], 1)) * 1/new_vals["convF"] * DSNU_var #fixed stipe pattern

    dsnu_row_rand_ratio = new_vals["dsnu_cont_sldr"]/100
    
    #load random dsnu only when necessary
    if dsnu_row_rand_ratio != 0:

        #Random Pixel DSNU
        rand_dsnu_frame = np.loadtxt("Resources/random_1024x1024.csv", delimiter = ',') #base file: 2D random distribution w/ stdv 1 around 0
        rand_dsnu_frame = rand_dsnu_frame[:frame_phots.shape[-1],:frame_phots.shape[-1]]
        rand_dsnu_frame = rand_dsnu_frame*1/new_vals["convF"] * DSNU_var

        #superimpose random and columnwise dsnu frames
        dsnu_frame_total = (rand_dsnu_frame * dsnu_row_rand_ratio**0.5) + (pattern_noise_frame * (1-dsnu_row_rand_ratio)**0.5)

    else:
        dsnu_frame_total = pattern_noise_frame

    frame_sim_fullres = rand_pG((frame_phots*qe_eff),new_vals)
    
    #frame_sim_fullres = frame_sim_fullres + np.round(pattern_noise_frame)
    frame_sim_fullres = frame_sim_fullres + np.round(dsnu_frame_total)

    frame_img = bin_array_sum(frame_sim_fullres, bin_fac(new_vals)) 
    
    #cut image data according to bit depth of the camera
    frame_img[frame_img>=2**new_vals["adc_bit"]] = int(2**new_vals["adc_bit"])-1

    ####Plot data ========================================================= 
    
    #Top left: Photon Map .........................................................
    truth_plot(frame_phots, new_vals)
    
    ##Top Right: Noisy Image data ....................................................    
    generate_sim_im_plot(frame_img, new_vals)   
    
    ###Bottom left: line plot ..................................................   
    generate_line_profile_plot(frame_img, frame_phots, new_vals)
        
    ####Bottom Right: histogramm .......................................................
    generate_histogram_plot(frame_img) 

    if export_choice == "Simulated Image as TIFF":
            safe_as_tiff(frame_img)

    return fig    

def calc_snr(phi=1, t_exp=1, bin_fac=0, qe=1,
                 pxl_pitch=6.5, ron=0, mu_d=0, fwc=100000):
        
        """
        Parameter:
            
            phi (float):    photon flux density in ph/(um)²/sec
            t_exp (float):  exposure_time in seconds
            bin_fac (int):  binning factor applied 2: 0=1x1; 1=2x2, 2=4x4
            
        """
        
        signal_e = qe * (phi*t_exp*pxl_pitch**2)
        n_dark = (mu_d*t_exp)**0.5
        n_shot = (signal_e)**0.5
        n_tot = (n_dark**2+n_shot**2+ron**2)**0.5
        snr = signal_e / n_tot
        
        return [snr, signal_e, n_tot, n_dark, n_shot, bin_fac]

def snr_info(input_vals):
    
    """
    Returns a string that provides info on the SNR according to current settings.
    """
    
    phi_pfd_max = input_vals["phi_pfd_max"]
    t_exp = input_vals["t_exp"]
    pxl_pitch = input_vals["pxl_pitch"]
    ron = input_vals["ron"]
    mu_dark = input_vals["mu_dark"]
    full_well_cap = input_vals["full_well_cap"]
    qe_eff = input_vals["qe_eff"]
    bin_factor = input_vals["bin_factor"]
    
    snr_data = calc_snr(phi_pfd_max, t_exp, bin_factor, qe_eff, 
                    pxl_pitch, ron, mu_dark, full_well_cap)

    my_snr_info = {"Photon Flux Density [ph/(um)²/sec]": phi_pfd_max,
                   "Quantum Efficiency [%]": qe_eff*100,
                   "Pixel Pitch [um]:" : pxl_pitch,
                   "Exposure Time [sec]:" :  t_exp,
                   "Binning Level" : bin_factor,
                   "---":0,
                " Signal-to-Noise Ratio SNR": snr_data[0],
                "SNR w/ binning": snr_data[0]*2**snr_data[5], 
                "Mean Signal [e-/pxl]": snr_data[1],
                "Total Noise [e-/pxl]": snr_data[2],
                "Dark Noise Contribution [e-/pxl]": snr_data[3],
                "Shot Noise Contribution [e-/pxl]": snr_data[4],
                "Read Noise Contribution [e-/pxl]": ron}
    
    return(my_snr_info)        

############################## PROGRAMM ###########################

values = make_sidebar()

# Create two columns
col1, col2 = st.columns([8,2])

with col2:
    st.image("Resources/EXCpco.png", width='stretch')

with col1:
    st.title("sCMOS Image Emulator")

st.info("""
        With this little tool you can superimpose an input image according to camera performance and 
        acquisition parameters such as exposure time, illumination situation and all major sources of noise
        in CMOS image sensors: read noise, photon noise and dark noise. Simply use the sidebar to make your settings and
        launch the calculator via the **Run Simulation** button.

        Histogram and line profile data are also given for illustrating the impact of different settings on the result image. 
        The outcome data can be stored either as PDF summary or 16-bit TIFF (image only). To safe the virtual camera image
        simply use the **DOWNLOAD** button, that appears  in the sidebar after pressing **Run Simulation**.
        """)

# Buttonpress for Simulation
launch_button = st.button("Run Simulation", width='stretch')

#Launch Simulation upon button press and...
st.subheader("Image Simulation Summary")

# Initialize storage for the figure if not already there
if "fig" not in st.session_state:
    st.session_state.fig = None

# Button to update the plot
if launch_button:
    #to ensure the program doesn't crash when a too small image is loaded...
    try:
        #try to run program
        display_figure = make_plots(values)
        st.session_state.fig = display_figure
    except:
        #...and display an exception warning text if an error is encountered!
        st.write("  :no_entry_sign: :x: **Ooops! Something went wrong! I can't generate an image! Please try different settings!** :x: :no_entry_sign:  ")
        st.session_state.fig = None
    

# Display the figure (only if one exists)
if st.session_state.fig is not None:
    st.pyplot(st.session_state.fig)

  
#...print some explanatory text
st.markdown("**Result:** Square shaped ROI for a virtual scientific CMOS camera. The output grayscale image is entirely simulated as a function " \
            "of the input or product specification data. Be aware that this tool cannot simulate real image data and is intended to be used for " \
            "educational purposes only! Its aim is illustrating how different datasheet parameters can influence our image quality. In addition, this "\
            "tool can serve as a nice assistance for determining a suitable camera for a given experiment.")

#st.subheader("Signal-to-Noise Performance")

#st.text("In the table below you find information regarding the signal-to-noise ratio under specified experiment conditions. "\
#         "Per definition, we assume HOMOGNEOUS illumination at the extent of specified max. photon flux density. For a " \
#         "sensible result keep the illumination strength within the cameras capabilities, i.e. the signal within the full "\
#         "well capacity of the sensor.")

# #...show a signal do noise ratio consideration
#df = pd.DataFrame.from_dict(
#    {k: f"{v:.4f}" for k, v in snr_info(values).items()},
#    orient="index", columns=["Value"])

#st.table(df)

st.markdown(
    """
    <div style="text-align: center; padding: 20px; color: gray; font-size: 14px;">
        © 2026 Excelitas Technologies
    </div>
    """,
    unsafe_allow_html=True
)


# quick debug
#st.write(values)




