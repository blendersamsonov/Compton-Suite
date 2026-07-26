import numpy as np

n = 1000
# RMS values: [sigma_x, sigma_xp, sigma_y, sigma_yp, sigma_z, sigma_dp]
sigmas = [1.598e-4, 1.598e-5, 1.598e-4, 1.598e-5, 6.0e-2, 2.5e-3]

# Generate 6D Gaussian data (zero mean, uncorrelated)
data = np.random.randn(n, 6) * sigmas

with open('distribution.ele', 'w') as f:
    f.write('''SDDS1
&description text="6D Gaussian electron beam. Energy=1 GeV, sigma_z=200 ps (RMS), delta=0.25% (RMS), eps_N=5 mm-mrad, beta=10 m." &
&parameter name=TotalParticles type=long description="Number of macroparticles" fix=1000 &
&parameter name=Energy type=double units=GeV description="Mean beam energy" fix=1.0 &
&parameter name=NormEmittance type=double units=mm*mrad description="Normalized emittance" fix=5.0 &
&parameter name=BetaFunction type=double units=m description="Beta function at location" fix=10.0 &
&parameter name=SigmaX type=double units=m description="RMS horizontal beam size" fix=1.598e-4 &
&parameter name=SigmaXp type=double units=rad description="RMS horizontal divergence" fix=1.598e-5 &
&parameter name=SigmaY type=double units=m description="RMS vertical beam size" fix=1.598e-4 &
&parameter name=SigmaYp type=double units=rad description="RMS vertical divergence" fix=1.598e-5 &
&parameter name=SigmaZ type=double units=m description="RMS bunch length" fix=6.0e-2 &
&parameter name=SigmaDp type=double description="RMS momentum spread" fix=2.5e-3 &
&column name=x type=double units=m description="Horizontal position" &
&column name=xp type=double units=rad description="Horizontal angle (px/pz)" &
&column name=y type=double units=m description="Vertical position" &
&column name=yp type=double units=rad description="Vertical angle (py/pz)" &
&column name=z type=double units=m description="Longitudinal position (head-tail, positive = ahead)" &
&column name=dP type=double description="Relative momentum deviation (p-p0)/p0" &
&data mode=ascii
''')
    for row in data:
        f.write(f" {row[0]:.6e}  {row[1]:.6e}  {row[2]:.6e}  {row[3]:.6e}  {row[4]:.6e}  {row[5]:.6e}\n")
    f.write("&end\n")

print("File 'distribution.ele' generated successfully with 1000 macroparticles.")
