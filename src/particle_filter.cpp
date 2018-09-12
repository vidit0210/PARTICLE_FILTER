
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    /* Set the number of particles, weights vector */
    num_particles = 100;

    for(int i = 0; i < num_particles; i ++)
    {
        weights.push_back(1.0);
    }

    /* Initialize all particles to first position.*/
    std::default_random_engine gen;
    std::normal_distribution<double> dist_x(x,std[0]);
    std::normal_distribution<double> dist_y(y,std[1]);
    std::normal_distribution<double> dist_theta(theta,std[2]);

    for(int i = 0; i < num_particles; i++)
    {
        Particle sample_particle;

        sample_particle.id = i;
        sample_particle.x = dist_x(gen);
        sample_particle.y = dist_y(gen);
        sample_particle.theta = dist_theta(gen);
        sample_particle.weight = 1.0;

        /* push the sample particle into the vector */
        particles.push_back(sample_particle);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    /* For the prediction formula about yaw rate is 0, ref to extended kalman filter 20. Sigma a point prediction assignment 1 in lecture 7 */

    /*
       NOTICE HERE
       The default_random_engine shall be out the bracket
       NOTICE HERE
     */
    std::default_random_engine gen;
    for(int i = 0; i < particles.size(); i ++)
    {
        double particle_x = particles[i].x;
        double particle_y = particles[i].y;
        double particle_theta = particles[i].theta;

        if(fabs(yaw_rate) < 0.00001)
        {
            particle_x += velocity * cos(particle_theta) * delta_t;
            particle_y += velocity * sin(particle_theta) * delta_t;

        } else
        {
            particle_x += velocity/yaw_rate * (sin(particle_theta + yaw_rate * delta_t) - sin(particle_theta));
            particle_y += velocity/yaw_rate * (cos(particle_theta) - cos(particle_theta + yaw_rate * delta_t));
            particle_theta += yaw_rate * delta_t;
        }
        /*
         NOTICE HERE
         The default_random_engine shall be out the bracket
         NOTICE HERE
         */
        //std::default_random_engine gen;
        std::normal_distribution<double> dist_x(particle_x,std_pos[0]);
        std::normal_distribution<double> dist_y(particle_y,std_pos[1]);
        std::normal_distribution<double> dist_theta(particle_theta,std_pos[2]);

        particle_x = dist_x(gen);
        particle_y = dist_y(gen);
        particle_theta = dist_theta(gen);

        particles[i].x = particle_x;
        particles[i].y = particle_y;
        particles[i].theta = particle_theta;

        //std::cout << "Prediction [" << i << "], x = " << particle_x << ", y = " << particle_y << ", theta = " << particle_theta << std::endl;
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    double dist_min;
    double dist_predict_obs;
    for(int i = 0; i < observations.size(); i ++)
    {
        dist_min = numeric_limits<double>::max();
        for(int j = 0; j < predicted.size(); j ++)
        {
            dist_predict_obs = dist(observations[i].x, observations[i].y,predicted[j].x, predicted[j].y);
            if(dist_predict_obs < dist_min)
            {
                dist_min = dist_predict_obs;
                observations[i].id = predicted[j].id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    for(int i = 0; i < num_particles; i ++)
    {
        /* The particle coordinate in map  */
        double x_p = particles[i].x;
        double y_p = particles[i].y;
        double theta_p = particles[i].theta;

        /* Note: Get the landmarks in the sensor range  */
        vector<LandmarkObs> Landmarks_in_ranage;
        for(int j = 0; j < map_landmarks.landmark_list.size(); j ++)
        {
            float x_lm = map_landmarks.landmark_list[j].x_f;
            float y_lm = map_landmarks.landmark_list[j].y_f;
            int id_lm = map_landmarks.landmark_list[j].id_i;

            LandmarkObs Landmark_in_map;
            Landmark_in_map.x = x_lm;
            Landmark_in_map.y = y_lm;
            Landmark_in_map.id = id_lm;

            if(dist(x_p, y_p, x_lm, y_lm) <= sensor_range)
            {
                Landmarks_in_ranage.push_back(Landmark_in_map);
            }
        }

        /* Transform observations to map coordinate system */
        vector<LandmarkObs> observations_map;
        for(int j = 0; j < observations.size(); j ++)
        {
            LandmarkObs observation_map;

            double x_c = observations[j].x;
            double y_c = observations[j].y;
            int id_c = observations[j].id;

            double x_m = x_p + cos(theta_p) * x_c - sin(theta_p) * y_c;
            double y_m = y_p + sin(theta_p) * x_c + cos(theta_p) * y_c;

            observation_map.x = x_m;
            observation_map.y = y_m;
            observation_map.id = id_c;
            observations_map.push_back(observation_map);
        }

        /* Data Association */
        dataAssociation(Landmarks_in_ranage, observations_map);

        /* Update weight */
        double std_x = std_landmark[0];
        double std_y = std_landmark[1];

        double weight = 1.0;
        particles[i].weight = weight;

        for(int j = 0; j < observations_map.size(); j ++)
        {
            double x_obs = observations_map[j].x;
            double y_obs = observations_map[j].y;
            double id_obs = observations_map[j].id;

            double mu_x = 0.0, mu_y = 0.0;
            for(int k = 0; k < Landmarks_in_ranage.size(); k ++)
            {
                if(id_obs == Landmarks_in_ranage[k].id)
                {
                    mu_x = Landmarks_in_ranage[k].x;
                    mu_y = Landmarks_in_ranage[k].y;
                    std::cout <<mu_x<<","<<mu_y<<"<==>"<<x_obs<<","<<y_obs<<std::endl;
                }
            }

            /* Calculate normalization term */
            double gauss_norm = 1.0/ (2.0 * M_PI * std_x * std_y);

            /* Calculate exponent */
            double exponent = (x_obs - mu_x) * (x_obs - mu_x) /(2.0 * std_x * std_x) + (y_obs - mu_y) * (y_obs - mu_y) /(2.0 * std_y * std_y);
            weight *= gauss_norm * exp(- exponent);
            particles[i].weight *= gauss_norm * exp(- exponent);
        }

        weights[i] = weight;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::default_random_engine gen;
    std::discrete_distribution<> dist_index(weights.begin(), weights.end());
    vector<Particle> resampleParticles;

    for(int i = 0; i < particles.size(); i ++)
    {
        int index = dist_index(gen);
        //std::cout << "Index for resample is " << index << std::endl;
        resampleParticles.push_back(particles[index]);
    }

    std::cout << "Resample particle num is : " << resampleParticles.size() << std::endl;
    particles = resampleParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                         const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}