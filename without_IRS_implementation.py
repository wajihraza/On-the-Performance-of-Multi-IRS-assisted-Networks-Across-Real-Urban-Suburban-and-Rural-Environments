from osgeo import ogr
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import multiprocessing as mp
from osgeo import ogr

#Assigning all values

c = 3*pow(10,8)
fc = 2.4*pow(10,9)
B = 20*10**6
a_L = 2
a_n = 3.3
x_uhf = 1
pt = 30
Pt_dB = 10*np.log10(pt) # transmitted power in dB
rate_threshold = 2e5
Pr_dB = np.array([])
u = 1
m = 1.0 #Shape for nakagami
omega = np.sqrt(0.5) #Param for nakagami
pr_los = []
pr_nlos = []
N1 = 512

bs_coords = [-2.790, 54.0080, 35]
irs_coords = [-2.784, 54.004, 20]
bs = bs_coords
irs = irs_coords

n_bs = 1
n_irs = 1

n_user = 400
N_val = [512] # number of IRS Elements
NF = 10
mu = 0
std_dev = np.sqrt(0.5)
K = 3
user_Long_range = [-2.794, -2.782]
user_Lat_range = [54.001, 54.014]


#FUNCTIONS
def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    # Radius of the Earth in kilometers
    radius = 6371.0
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    # Calculate distance
    distance = radius * c
    return distance*1000
# Function to check LOS with obstacles
def has_los_with_obstacles_1(p1, p2, obstacles=[]):
    line = LineString([p1, p2])
    for obstacle in obstacles:
        if line.intersects(obstacle):
            return False
    return True

# Function to check LOS with obstacles and building heights using grid-based approach
def has_los_with_obstacles(p1, p2, obstacles=[], grid_resolution=50):
    x_vals = np.linspace(p1[0], p2[0], grid_resolution)
    y_vals = np.linspace(p1[1], p2[1], grid_resolution)
    z_vals = np.linspace(p1[2], p2[2], grid_resolution)

    for i in range(grid_resolution):
        point = (x_vals[i], y_vals[i], z_vals[i])

        for obstacle in obstacles:
            if obstacle.contains(Point(point[:2])):
                obstacle_row = obstacle_gdf[obstacle_gdf['geometry'] == obstacle]
                obstacle_height = obstacle_row.iloc[0]['height']
                if point[2] <= obstacle_height:
                    return False
                break  # No need to check other obstacles if one is found
    return True

def compute_pathloss(distance,path_loss_exponent):
  y = c/fc
  path_loss = 20*np.log10(4*np.pi/y)+10*path_loss_exponent*np.log10(distance)+x_uhf
  path_loss = np.array(path_loss)
  return path_loss

def generate_nakagami_samples(m, omega, size):
    magnitude_samples = np.sqrt(omega) * np.sqrt(np.random.gamma(m, 1, size)) / np.sqrt(np.random.gamma(m - 0.5, 1, size))
    phase_samples = np.random.uniform(0, 2 * np.pi, size=size)
    complex_samples = magnitude_samples * np.exp(1j * phase_samples)
    return complex_samples

def compute_distances(user_positions, base_stations):
    distances = np.sqrt(np.sum(np.square(user_positions - base_stations), axis=1))
    return distances

def generate_rayleigh_fading_channel(num_users, std_mean, std_dev):
    X = np.random.normal(std_mean, std_dev, num_users)
    Y = np.random.normal(std_mean, std_dev, num_users)
    rayleigh_channel = np.abs(X + 1j*Y)
    return rayleigh_channel

def rician_fading_channel(num_samples, K, sigma):
    gaussian_samples = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)
    # Generate Rician fading channel samples
    h = np.sqrt(K / (K + 1)) * (gaussian_samples + np.sqrt(1 / (2 * (K + 1))) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)))
    noise = sigma * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    h_with_noise = h + noise
    return abs(h_with_noise)

# Function to compute outage probability at each iteration
def compute_outage_probability(num_users, rate, rate_threshold):
    outage = 0
    for j in range(num_users):
        if rate[j] < rate_threshold:
            outage += 1
    return outage / num_users

def compute_SNR(PR, noise_floor):
    SNR = PR - noise_floor
    return SNR

def compute_rate(SNR):
    SNR_watts = (10**(SNR/10))
    
    if len(SNR) == 0:
        x = 1
    else:
        x = len(SNR)
    return (B/(n_bs*x))*np.log2(1 + SNR_watts)

def total_coverage_probability(rate_mat, threshold_rate):
    coverage_probability = np.sum(rate_mat > threshold_rate) / n_user
    return coverage_probability

def find_common_locations(locations1, locations2):
    common_locations = []
    for loc1 in locations1:
        for loc2 in locations2:
            if not np.array_equal(loc1, [0, 0]) and not np.array_equal(loc2, [0, 0]) and np.all(loc1 == loc2):
                common_locations.append(tuple(loc1))
                break  # Once found, no need to continue checking
    return common_locations

def replace_common_locations(channel, common):
    new_channel = []
    for loc in channel:
        if tuple(loc) in common:
            new_channel.append([0, 0])
        else:
            new_channel.append(loc)
    return new_channel

def coverage_probability(rate_mat, threshold_rate, n_user):
    coverage_probability = np.sum(rate_mat > threshold_rate) / n_user
    return coverage_probability



# Load the shapefile into a GeoDataFrame
shapefile_path = "C:\\Visual Studio code\\Test\\output_shapefile_lancaster.shp"
gdf = gpd.read_file(shapefile_path)


# Iterate through each base station
# Read obstacles from a shapefile
obstacle_shapefile = "C:\\Visual Studio code\\Test\\output_shapefile_lancaster.shp"
obstacle_gdf = gpd.read_file(obstacle_shapefile)
# Define base station and user coordinates
user_longitudes = np.random.uniform(user_Long_range[0], user_Long_range[1], n_user)
user_latitudes = np.random.uniform(user_Lat_range[0], user_Lat_range[1], n_user)
user_coords = list(zip(user_longitudes, user_latitudes))

# Create empty lists for LOS and NLOS users
all_los_users_x = []
all_los_users_y = []
all_nlos_users_x = []
all_nlos_users_y = []


los_users = []
nlos_users = []



# Scatter plot for base stations
bs_longitudes = [bs_coords[0]]
bs_latitudes = [bs_coords[1]]

# Arrays to store LOS and NLOS users for each BS
los_users_per_bs = [[] for _ in range(n_bs)]
nlos_users_per_bs = [[] for _ in range(n_bs)]



# Read obstacles from a shapefile
obstacle_shapefile = "C:\\Visual Studio code\\Test\\output_shapefile_lancaster.shp"
obstacle_gdf = gpd.read_file(obstacle_shapefile)


# Create empty lists for user markers
los_users = []          # LOS User from bs
irs_los_users = []      # IRS-LOS User
irs_nlos_users = []     # IRS-NLOS User
other_users = []        # Remaining Users

# Iterate through each user
for user_coord in user_coords:
    user_point = Point(user_coord)
    los_flag = False
    irs_los_flag = False
    irs_nlos_flag = False

    # Check LOS between each base station and user


# Read obstacles from a shapefile
obstacle_shapefile = "C:\\Visual Studio code\\Test\\output_shapefile_lancaster.shp"
obstacle_gdf = gpd.read_file(obstacle_shapefile)
# Add random heights to the GeoDataFrame
random_obstacle_heights = np.random.uniform(10, 17, len(obstacle_gdf))
obstacle_gdf['height'] = random_obstacle_heights

user_longitudes = np.random.uniform(user_Long_range[0], user_Long_range[1], n_user)
user_latitudes = np.random.uniform(user_Lat_range[0], user_Lat_range[1], n_user)
user_heights = np.random.uniform(0,0.5, n_user)  # Random user heights
user_coords = list(zip(user_longitudes, user_latitudes, user_heights))

# Create empty lists for user markers
los_users = []          # LOS User from bs
irs_los_users = []      # IRS-LOS User
irs_nlos_users = []     # IRS-NLOS User
other_users = []        # Remaining Users
has_los_bs_irs = []

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def process_coverage2(n_user_idx,  n_user, n_iterations):
    snr = []
    N1 = 256
    final_rate = []
    for _ in range(n_iterations):
        user_longitudes = np.random.uniform(user_Long_range[0], user_Long_range[1], n_user)
        user_latitudes = np.random.uniform(user_Lat_range[0], user_Lat_range[1], n_user)
        user_heights = np.random.uniform(0,0.5, n_user)  # Random user heights
        user_coords = list(zip(user_longitudes, user_latitudes, user_heights))
        los_users = []
        nlos_users = []

        for user_coord in user_coords:
            user_point = Point(user_coord[:2])
            user_height = user_coord[2]

            has_los = has_los_with_obstacles((bs[0],bs[1],bs[2]), (user_point.x, user_point.y, user_height), obstacle_gdf['geometry'])

            if has_los:
                los_users.append(user_coord)
            else:
                nlos_users.append(user_coord)


        #Measuring distances for all
        los_users = np.array(los_users)

        #Distance for directly los users from base station to users
        dist_los_users = haversine_distance([user[1] for user in los_users], [user[0] for user in los_users], bs[1], bs[0])
        dist_los_users = np.sqrt(dist_los_users**2+(los_users[:,2]-bs[2])**2)

        #Distance for non line sight user from base station to nlos users
        dist_nlos_users = haversine_distance([user[1] for user in nlos_users], [user[0] for user in nlos_users], bs[1], bs[0])
        dist_nlos_users = np.sqrt(dist_nlos_users**2+(bs[2])**2)

        #------------------------------------------------------------------------------------------------------------------/////////////////
        # Rayleigh Fading Channel for los user
        g_m = rician_fading_channel(len(los_users), K, omega)
        g_m = g_m.reshape(1,-1)

        los_dist = dist_los_users
        los_pathloss = compute_pathloss(los_dist,a_L)
        los_pathloss = los_pathloss.reshape(1,-1)
        Pr_los = []
        for pl_los in los_pathloss:
            Pr_los = np.append(Pr_los, Pt_dB + 10*np.log10(g_m**2)- pl_los ) # received power for LOS links
        Pr_los = Pr_los.reshape(1,-1)

        #------------------------------------------------------------------------------------------------------------------/////////////////
        # Rayleigh Fading Channel for nlos user

        c_m = generate_rayleigh_fading_channel(len(nlos_users), mu, std_dev)
        c_m = c_m.reshape(1,-1)
        nlos_dist = dist_nlos_users
        nlos_pathloss = compute_pathloss(nlos_dist,a_n)
        nlos_pathloss = nlos_pathloss.reshape(1,-1)
        Pr_nlos = []
        for pl_nlos in nlos_pathloss:
            Pr_nlos = np.append(Pr_nlos, Pt_dB + 10*np.log10(c_m**2)- pl_nlos ) # received power for LOS links
        Pr_nlos = Pr_nlos.reshape(1,-1)

        Pr_dB = np.concatenate((Pr_los, Pr_nlos),axis = None)
        np.random.shuffle(Pr_dB)
        Pr_dB = Pr_dB.reshape(-1)

        noise = -174+10*np.log10(B) + NF
        SNR_matrix = []

        # Calculate SNR
        SNR = compute_SNR(Pr_dB, noise)
        SNR_matrix.append(SNR)

        SNR_matrix = np.array(SNR_matrix)
        SNR_matrix = SNR_matrix.reshape(1,n_user)
            # Calculate rate
        rate_matrix = compute_rate(SNR_matrix)
        rate_matrix = rate_matrix.reshape(-1)
        final_rate.append(rate_matrix)
    rate = np.mean(final_rate, axis=0)

    return rate
    


# Define simulation parameters
n_iterations = 10  # Number of iterations
n_user_vals = np.linspace(50,500,50).astype(int)  # Values of n_user
rate_per_unit_area = np.zeros((len(n_user_vals)),dtype=object)
for n_user_idx, n_user in enumerate(n_user_vals):
    print(n_user_idx)
    p1 = process_coverage2(n_user_idx,  n_user, n_iterations) 
    rate_per_unit_area[n_user_idx] = p1

print("Output Shape: ", np.shape(rate_per_unit_area))
print("Output is ", rate_per_unit_area)
np.save('x_code_0irs.npy',rate_per_unit_area)

