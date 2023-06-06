import numpy as np
import matplotlib.pyplot as plt

def generate_data_wlan_channel(num_users, num_samples,rng=None):
    '''
        This code generates channel gains and shadowing values for each user, 
        computes the received power and noise for each user, 
        adds Rayleigh fading to the signal-to-noise ratios (SNRs), 
        and then normalizes the SNRs by the maximum value to generate rewards.

        Author: Abhishek Sinha 
        Converted from Matlab to python using chatGPT
    '''

    if rng is None:
        rng = np.random.default_rng()

    # Define simulation parameters
    # num_users = 5;

    # Define simulation parameters
    tx_power = 20
    noise_floor = -100
    channel_gain_mean = 0#30*(np.arange(num_users)[:,np.newaxis]+1)
    channel_gain_var = 5
    shadowing_mean = 0
    shadowing_var = 8#np.random.randint(1, 10, (num_users, 1))#8
    freq = 5e9
    speed = 10
    sampling_time = 0.001
    # num_samples = 10000;

    # Generate random channel gains for each user
    # channel_gains = channel_gain_mean + channel_gain_var * np.random.randn(num_users, 1)
    channel_gains = channel_gain_mean + channel_gain_var * rng.standard_normal(size=(num_users, 1))

    # print(f'channel_gains {channel_gains.shape}')

    # Generate random shadowing for each user
    # shadowings = shadowing_mean + shadowing_var * np.random.randn(num_users, num_samples)
    shadowings = shadowing_mean + shadowing_var * rng.standard_normal(size=(num_users, num_samples))

    # Compute the received power and noise for each user
    # distances = np.random.randint(1, 101, (num_users, 1))
    distances = rng.integers(1, 101, (num_users, 1))
    path_loss = 20*np.log10(4*np.pi*distances*freq/3e8)
    received_powers = tx_power - channel_gains - path_loss - shadowings
    noises = noise_floor * np.ones((num_users, num_samples))

    # print(f'shadowings {shadowings.shape}')
    # print(f'distances {distances.shape}')
    # print(f'path_loss {path_loss.shape}')
    # print(f'received_powers {received_powers.shape}')
    # print(f'noises {noises.shape}')

    # Compute the instantaneous SNR for each user
    snrs = received_powers - noises

    # Add Rayleigh fading to the SNRs
    t = np.arange(num_samples) * sampling_time
    # fading = np.sqrt(2/np.pi) * (np.random.randn(num_users, num_samples) + 1j*np.random.randn(num_users, num_samples)) * np.exp(-1j*2*np.pi*speed*t*np.cos(np.deg2rad(np.random.randint(-180, 181, (num_users, 1)))))
    fading = (np.sqrt(2/np.pi) * 
              (rng.standard_normal((num_users, num_samples)) + 1j*rng.standard_normal((num_users, num_samples))) *
              np.exp(-1j*2*np.pi*speed*t*np.cos(np.deg2rad(rng.integers(-180, 181, (num_users, 1))))))
    snrs = snrs + 20*np.log10(np.abs(fading))
    # print(f'noises {noises.shape}')

    # Normalize the SNRs and return as rewards
    # Rewards need to be positive so we shift the SNRs slightly if there are any 
    # negative SNR values
    if np.min(snrs)<=0:
        snrs = snrs + np.abs(np.min(snrs))+0.001
    normalization = np.max(snrs)
    rewards = (snrs/normalization)

    # Plot the Rewards
    for i in range(num_users):
        plt.plot(t[:500], rewards[i,:500],label=f'user {i+1}',linewidth=1.0)
    plt.xlabel('Time (s)')
    plt.legend()
    plt.ylabel('Instantaneous SNR (dB)')
    return rewards