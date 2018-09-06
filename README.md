# Wind power forecast example

These netCDF files contain data from the grid node closest to a Chilean wind farm from three global weather models, as well as
from anemometers on-site and total wind farm production data from the turbines' SCADA system. The wind farm consists of over 20 turbines
with a hub height of 80 m.


## Data description

### NWP data

The data contain forecast runs for the given parameters starting a the POSIX timestamp (seconds after 1970-01-01 00:00) given by
`epoch_s`, with a forecast offset in seconds given by `offset_s`.

### Met mast data

Quantities `ws80m` and `ws34m` are taken from a met mast located inside the wind farm. The other quantities are averages over all available
nacelle anemometers, and as such may not be as accurate or quality controlled.

### Wind power production data

Power production is normalized to the farm capacity and thus ranges from 0 to 1. The `rawpower` is what a met service provider would
usually receive as feed-in data from the neighboring power grid station. Here we have no information about turbine maintainance and
failures, as well as curtailment measures. In case of this farm however, we received status information and production info from all individual
turbines. In addition to taking these into account, we also applied a Gaussian Process based filtering procedure to remove additional outliers.
The result is compiled into `power`.


## Questions, Tasks, and Caveats


Your task is to eventually produce a wind power forecast every three hours for up to 4 days ahead for the given wind farm. The idea is to train a
DNN with (some of the) met data as input, and power produced in the next 4 days as output. You have been provided the data
as-is from your customer. Assume for now you have not been provided the `power` data, only `rawpower`.

1. Take a look at all data files and find a common time range envelope you can use for training (e.g. use `pandas.Timstamp(epoch_s * 1e9)` or
`epoch_s.astype('datetime64[s]')` to get actual timestamps to work with).
2. You may also want to consider selecting a suitable test set. (Caveat: Make sure to select contiguous time ranges over at least a week,
because you are forecasting for several days.) Ideally, it should be the same set for all experiment, such that you can more easily compare
different setups. What criteria can you think of for a good test set?
3. You target values could probably use some cleaning. Try plotting wind speed vs. power prduction. This gives you the so-called power curve of
the wind farm. Look up `wind farm power curve` on the internet. Due to the finite turbine response time and rotor area, you will see a lot of
scatter and there is a whole branch of the wind power research community working on how to get realistic power curves from data. Our focus here
is however to weed out obvious outliers, nothing more. Can you think of a way to do so?
4. Now construct a training data set from the NWP data, using the cleaned power data as targets. For starters, let's use all parameters provided.
Try with one, two and all three weather models as input (Hint: One or two hidden layers with around 32 neurons should be enough. Make sure your
validation data is also comprised of blocks of at least a few days length).
Plot the RMSE of output vs. target over the forecast horizon. Does the result improve with more models? Can you identify the model
contributing the most to the solution?
5. Look at the first layer weights of your DNNs. Can you identify parameters that do not seem to contribute? Does the result improve if you
leave them out?
6. Now add the met mast data to the mix, assuming you can access it online with no delay. This should improve your forecasts in the first few hours.
7. But wait a minute - is this realistic? After all, the NWPs take a while to compute and download. We are talking about a delay of maybe 6 h
after the analysis time until you can use them in an operational setting. How do you have to modify your setup to reflect that?
8. Do you observe the error decreasing dramatically in the first forecast hours? If so, you setup may now suffer from the fact that the errors at
the later forecast horizons are much larger, and thus get optimized first by the training algorithm. Considering you do not have nearly enough
data to capture all the complexity of the problem, maybe you could ease the load on your ML algorithms by splitting the forecast range into
intervals and train them separately?
9. Does the forecast reassembled from several intervals perform better than the all-in-one forecast? Do the individual forecasts look choppy?
Maybe the intervals should overlap?
10. Assume you also have current feed-in data from the wind farm, i.e. you can use the power production data up to "now" in the same way as
the met data. Does this improve the result?
11. Now replace the `rawpower` data with the `power` data. Do the results improve? What happens if you train with `power`, then evaluate the
model on `rawpower`?
12. Assume you do not have historical production data for training purposes, only the met data. The customer also furnishes you with production
data for the test set and a tabulated power curve (which in this tutorial you have to construct yourself). How does the performance of the
models change if you train a DNN to forecast the measured wind speed, then pass the DNN output through the power curve and evaluate it against
the measured power on the test set?
