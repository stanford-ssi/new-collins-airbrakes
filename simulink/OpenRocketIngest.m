%Initial Open Rocket Data

T = readtable("AirbrakeSims.csv");

velocity = T.("VerticalVelocity_m_s_");
altitude = T.("Altitude_m_");
time = T.("x_Time_s_");
accel = T.("VerticalAcceleration_m_s__");
mass = T.("Mass_kg_");
rho = T.("AirDensity_kg_m__");
speed_of_sound = T.("SpeedOfSound_m_s_");

mach_threashold = 0.7;

sim_start_index = find(altitude > 6000, 1, 'first');

i = sim_start_index;

% while velocity(i) > speed_of_sound(i) * mach_threashold
%     i=i+1;
% end
% 
V_initial_2 = velocity(i)
altitude_initial = altitude(i)
mass_burnout = mass(i)

% CD Lookup Table
T_cd = readtable("MachNumberCDMap.csv");

% delete any rows where MachNumber___ < 0.1
T_cd(T_cd.("MachNumber___") < 0.1, :) = [];


mach_number = T_cd.("MachNumber___");
c_d = T_cd.("DragCoefficient___");

[mach_number, ia] = unique(mach_number,'stable');
c_d = c_d(ia);

[mach_number, idx] = sort(mach_number); c_d = c_d(idx);
