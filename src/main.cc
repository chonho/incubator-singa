/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#include "./singa.h"
/**
 * \file main.cc provides an example main function.
 *
 * Like the main func of Hadoop, it prepares the job configuration and submit it
 * to the Driver which starts the training.
 *
 * Users can define their own main func to prepare the job configuration in
 * different ways other than reading it from a configuration file. But the main
 * func must call Driver::Init at the beginning, and pass the job configuration
 * and resume option to the Driver for job submission.
 *
 * Optionally, users can register their own implemented subclasses of Layer,
 * Updater, etc. through the registration function provided by the Driver.
 *
 * Users must pass at least one argument to the singa-run.sh, i.e., the job
 * configuration file which includes the cluster topology setting. Other fields
 * e.g, neuralnet, updater can be configured in main.cc.
 *
 * TODO
 * Add helper functions for users to generate configurations for popular models
 * easily, e.g., MLP(layer1_size, layer2_size, tanh, loss);
 */
int main(int argc, char **argv) {

  // must create driver at the beginning and call its Init method.
  singa::Driver driver;
  driver.Init(argc, argv);

  // if -resume in argument list, set resume to true; otherwise false
  int resume_pos = singa::ArgPos(argc, argv, "-resume");
  bool resume = (resume_pos != -1);

  // users can register new subclasses of layer, updater, etc.

  // constract model and generate jobproto
  // clee

  singa::JobProto jobConf;
  singa::Builder builder(&jobConf);
  builder.CONV_ex3();
  //builder.MLP_ex2();
  //builder.Construct();
  builder.Display();

  // get the job conf, and custmize it if need
  //singa::JobProto jobConf = driver.job_conf();

  // submit the job for training
  driver.Train(resume, jobConf);

  return 0;
}
