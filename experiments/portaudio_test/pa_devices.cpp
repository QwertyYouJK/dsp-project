// pa_list_devices.cpp
// Build: g++ -O2 -std=c++17 pa_list_devices.cpp -lportaudio -o pa_list_devices
// Run:   ./pa_list_devices

#include <iostream>
#include <portaudio.h>

int main() {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "Pa_Initialize failed: " << Pa_GetErrorText(err) << " (" << err << ")\n";
        return 1;
    }

    int n = Pa_GetDeviceCount();
    if (n < 0) {
        std::cerr << "Pa_GetDeviceCount failed: " << Pa_GetErrorText(n) << " (" << n << ")\n";
        Pa_Terminate();
        return 1;
    }

    int defIn  = Pa_GetDefaultInputDevice();
    int defOut = Pa_GetDefaultOutputDevice();

    std::cout << "PortAudio device list (" << n << " devices)\n";
    std::cout << "Default IN  = " << defIn  << "\n";
    std::cout << "Default OUT = " << defOut << "\n\n";

    for (int i = 0; i < n; i++) {
        const PaDeviceInfo* di = Pa_GetDeviceInfo(i);
        if (!di) continue;
        const PaHostApiInfo* hai = Pa_GetHostApiInfo(di->hostApi);

        std::cout
            << "[" << i << "] "
            << (hai ? hai->name : "HostAPI?") << " | "
            << di->name
            << " | in=" << di->maxInputChannels
            << " out=" << di->maxOutputChannels
            << " | defSR=" << di->defaultSampleRate;

        if (i == defIn)  std::cout << "  (default IN)";
        if (i == defOut) std::cout << "  (default OUT)";
        std::cout << "\n";
    }

    Pa_Terminate();
    return 0;
}
