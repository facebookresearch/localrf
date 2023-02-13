#!/usr/bin/env python

import torch

import cupy
import re

kernel_Softsplat_updateOutput = '''
	extern "C" __global__ void kernel_Softsplat_updateOutput(
		const int n,
		const float* input,
		const float* flow,
		float* output
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		const int intN = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
		const int intC = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
		const int intY = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
		const int intX = ( intIndex                                                    ) % SIZE_3(output);

		float fltOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
		float fltOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);

		int intNorthwestX = (int) (floor(fltOutputX));
		int intNorthwestY = (int) (floor(fltOutputY));
		int intNortheastX = intNorthwestX + 1;
		int intNortheastY = intNorthwestY;
		int intSouthwestX = intNorthwestX;
		int intSouthwestY = intNorthwestY + 1;
		int intSoutheastX = intNorthwestX + 1;
		int intSoutheastY = intNorthwestY + 1;

		float fltNorthwest = ((float) (intSoutheastX) - fltOutputX   ) * ((float) (intSoutheastY) - fltOutputY   );
		float fltNortheast = (fltOutputX    - (float) (intSouthwestX)) * ((float) (intSouthwestY) - fltOutputY   );
		float fltSouthwest = ((float) (intNortheastX) - fltOutputX   ) * (fltOutputY    - (float) (intNortheastY));
		float fltSoutheast = (fltOutputX    - (float) (intNorthwestX)) * (fltOutputY    - (float) (intNorthwestY));

		if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(output)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(output))) {
			atomicAdd(&output[OFFSET_4(output, intN, intC, intNorthwestY, intNorthwestX)], VALUE_4(input, intN, intC, intY, intX) * fltNorthwest);
		}

		if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(output)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(output))) {
			atomicAdd(&output[OFFSET_4(output, intN, intC, intNortheastY, intNortheastX)], VALUE_4(input, intN, intC, intY, intX) * fltNortheast);
		}

		if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(output)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(output))) {
			atomicAdd(&output[OFFSET_4(output, intN, intC, intSouthwestY, intSouthwestX)], VALUE_4(input, intN, intC, intY, intX) * fltSouthwest);
		}

		if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(output)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(output))) {
			atomicAdd(&output[OFFSET_4(output, intN, intC, intSoutheastY, intSoutheastX)], VALUE_4(input, intN, intC, intY, intX) * fltSoutheast);
		}
	} }
'''

kernel_Softsplat_updateGradInput = '''
	extern "C" __global__ void kernel_Softsplat_updateGradInput(
		const int n,
		const float* input,
		const float* flow,
		const float* gradOutput,
		float* gradInput,
		float* gradFlow
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		const int intN = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput) / SIZE_1(gradInput) ) % SIZE_0(gradInput);
		const int intC = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput)                     ) % SIZE_1(gradInput);
		const int intY = ( intIndex / SIZE_3(gradInput)                                         ) % SIZE_2(gradInput);
		const int intX = ( intIndex                                                             ) % SIZE_3(gradInput);

		float fltGradInput = 0.0;

		float fltOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
		float fltOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);

		int intNorthwestX = (int) (floor(fltOutputX));
		int intNorthwestY = (int) (floor(fltOutputY));
		int intNortheastX = intNorthwestX + 1;
		int intNortheastY = intNorthwestY;
		int intSouthwestX = intNorthwestX;
		int intSouthwestY = intNorthwestY + 1;
		int intSoutheastX = intNorthwestX + 1;
		int intSoutheastY = intNorthwestY + 1;

		float fltNorthwest = ((float) (intSoutheastX) - fltOutputX   ) * ((float) (intSoutheastY) - fltOutputY   );
		float fltNortheast = (fltOutputX    - (float) (intSouthwestX)) * ((float) (intSouthwestY) - fltOutputY   );
		float fltSouthwest = ((float) (intNortheastX) - fltOutputX   ) * (fltOutputY    - (float) (intNortheastY));
		float fltSoutheast = (fltOutputX    - (float) (intNorthwestX)) * (fltOutputY    - (float) (intNorthwestY));

		if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(gradOutput)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(gradOutput))) {
			fltGradInput += VALUE_4(gradOutput, intN, intC, intNorthwestY, intNorthwestX) * fltNorthwest;
		}

		if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(gradOutput)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(gradOutput))) {
			fltGradInput += VALUE_4(gradOutput, intN, intC, intNortheastY, intNortheastX) * fltNortheast;
		}

		if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(gradOutput)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(gradOutput))) {
			fltGradInput += VALUE_4(gradOutput, intN, intC, intSouthwestY, intSouthwestX) * fltSouthwest;
		}

		if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(gradOutput)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(gradOutput))) {
			fltGradInput += VALUE_4(gradOutput, intN, intC, intSoutheastY, intSoutheastX) * fltSoutheast;
		}

		gradInput[intIndex] = fltGradInput;
	} }
'''

kernel_Softsplat_updateGradFlow = '''
	extern "C" __global__ void kernel_Softsplat_updateGradFlow(
		const int n,
		const float* input,
		const float* flow,
		const float* gradOutput,
		float* gradInput,
		float* gradFlow
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		float fltGradFlow = 0.0;

		const int intN = ( intIndex / SIZE_3(gradFlow) / SIZE_2(gradFlow) / SIZE_1(gradFlow) ) % SIZE_0(gradFlow);
		const int intC = ( intIndex / SIZE_3(gradFlow) / SIZE_2(gradFlow)                    ) % SIZE_1(gradFlow);
		const int intY = ( intIndex / SIZE_3(gradFlow)                                       ) % SIZE_2(gradFlow);
		const int intX = ( intIndex                                                          ) % SIZE_3(gradFlow);

		float fltOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
		float fltOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);

		int intNorthwestX = (int) (floor(fltOutputX));
		int intNorthwestY = (int) (floor(fltOutputY));
		int intNortheastX = intNorthwestX + 1;
		int intNortheastY = intNorthwestY;
		int intSouthwestX = intNorthwestX;
		int intSouthwestY = intNorthwestY + 1;
		int intSoutheastX = intNorthwestX + 1;
		int intSoutheastY = intNorthwestY + 1;

		float fltNorthwest = 0.0;
		float fltNortheast = 0.0;
		float fltSouthwest = 0.0;
		float fltSoutheast = 0.0;

		if (intC == 0) {
			fltNorthwest = ((float) (-1.0)) * ((float) (intSoutheastY) - fltOutputY   );
			fltNortheast = ((float) (+1.0)) * ((float) (intSouthwestY) - fltOutputY   );
			fltSouthwest = ((float) (-1.0)) * (fltOutputY    - (float) (intNortheastY));
			fltSoutheast = ((float) (+1.0)) * (fltOutputY    - (float) (intNorthwestY));

		} else if (intC == 1) {
			fltNorthwest = ((float) (intSoutheastX) - fltOutputX   ) * ((float) (-1.0));
			fltNortheast = (fltOutputX    - (float) (intSouthwestX)) * ((float) (-1.0));
			fltSouthwest = ((float) (intNortheastX) - fltOutputX   ) * ((float) (+1.0));
			fltSoutheast = (fltOutputX    - (float) (intNorthwestX)) * ((float) (+1.0));

		}

		for (int intChannel = 0; intChannel < SIZE_1(gradOutput); intChannel += 1) {
			float fltInput = VALUE_4(input, intN, intChannel, intY, intX);

			if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(gradOutput)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(gradOutput))) {
				fltGradFlow += fltInput * VALUE_4(gradOutput, intN, intChannel, intNorthwestY, intNorthwestX) * fltNorthwest;
			}

			if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(gradOutput)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(gradOutput))) {
				fltGradFlow += fltInput * VALUE_4(gradOutput, intN, intChannel, intNortheastY, intNortheastX) * fltNortheast;
			}

			if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(gradOutput)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(gradOutput))) {
				fltGradFlow += fltInput * VALUE_4(gradOutput, intN, intChannel, intSouthwestY, intSouthwestX) * fltSouthwest;
			}

			if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(gradOutput)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(gradOutput))) {
				fltGradFlow += fltInput * VALUE_4(gradOutput, intN, intChannel, intSoutheastY, intSoutheastX) * fltSoutheast;
			}
		}

		gradFlow[intIndex] = fltGradFlow;
	} }
'''

def cupy_kernel(strFunction, objVariables):
	strKernel = globals()[strFunction]

	while True:
		objMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

		if objMatch is None:
			break
		# end

		intArg = int(objMatch.group(2))

		strTensor = objMatch.group(4)
		intSizes = objVariables[strTensor].size()

		strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))
	# end

	while True:
		objMatch = re.search('(OFFSET_)([0-4])(\()([^\)]+)(\))', strKernel)

		if objMatch is None:
			break
		# end

		intArgs = int(objMatch.group(2))
		strArgs = objMatch.group(4).split(',')

		strTensor = strArgs[0]
		intStrides = objVariables[strTensor].stride()
		strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

		strKernel = strKernel.replace(objMatch.group(0), '(' + str.join('+', strIndex) + ')')
	# end

	while True:
		objMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

		if objMatch is None:
			break
		# end

		intArgs = int(objMatch.group(2))
		strArgs = objMatch.group(4).split(',')

		strTensor = strArgs[0]
		intStrides = objVariables[strTensor].stride()
		strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

		strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
	# end

	return strKernel
# end

@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
	return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
# end

class _FunctionSoftsplat(torch.autograd.Function):
	@staticmethod
	def forward(self, input, flow):
		self.save_for_backward(input, flow)

		intSamples = input.shape[0]
		intInputDepth, intInputHeight, intInputWidth = input.shape[1], input.shape[2], input.shape[3]
		intFlowDepth, intFlowHeight, intFlowWidth = flow.shape[1], flow.shape[2], flow.shape[3]

		assert(intFlowDepth == 2)
		assert(intInputHeight == intFlowHeight)
		assert(intInputWidth == intFlowWidth)

		assert(input.is_contiguous() == True)
		assert(flow.is_contiguous() == True)

		output = input.new_zeros([ intSamples, intInputDepth, intInputHeight, intInputWidth ])

		if input.is_cuda == True:
			n = output.nelement()
			cupy_launch('kernel_Softsplat_updateOutput', cupy_kernel('kernel_Softsplat_updateOutput', {
				'input': input,
				'flow': flow,
				'output': output
			}))(
				grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ n, input.data_ptr(), flow.data_ptr(), output.data_ptr() ]
			)

		elif input.is_cuda == False:
			raise NotImplementedError()

		# end

		return output
	# end

	@staticmethod
	def backward(self, gradOutput):
		input, flow = self.saved_tensors

		intSamples = input.shape[0]
		intInputDepth, intInputHeight, intInputWidth = input.shape[1], input.shape[2], input.shape[3]
		intFlowDepth, intFlowHeight, intFlowWidth = flow.shape[1], flow.shape[2], flow.shape[3]

		assert(intFlowDepth == 2)
		assert(intInputHeight == intFlowHeight)
		assert(intInputWidth == intFlowWidth)

		assert(gradOutput.is_contiguous() == True)

		gradInput = input.new_zeros([ intSamples, intInputDepth, intInputHeight, intInputWidth ]) if self.needs_input_grad[0] == True else None
		gradFlow = input.new_zeros([ intSamples, intFlowDepth, intFlowHeight, intFlowWidth ]) if self.needs_input_grad[1] == True else None

		if input.is_cuda == True:
			if gradInput is not None:
				n = gradInput.nelement()
				cupy_launch('kernel_Softsplat_updateGradInput', cupy_kernel('kernel_Softsplat_updateGradInput', {
					'input': input,
					'flow': flow,
					'gradOutput': gradOutput,
					'gradInput': gradInput,
					'gradFlow': gradFlow
				}))(
					grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
					block=tuple([ 512, 1, 1 ]),
					args=[ n, input.data_ptr(), flow.data_ptr(), gradOutput.data_ptr(), gradInput.data_ptr(), None ]
				)
			# end

			if gradFlow is not None:
				n = gradFlow.nelement()
				cupy_launch('kernel_Softsplat_updateGradFlow', cupy_kernel('kernel_Softsplat_updateGradFlow', {
					'input': input,
					'flow': flow,
					'gradOutput': gradOutput,
					'gradInput': gradInput,
					'gradFlow': gradFlow
				}))(
					grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
					block=tuple([ 512, 1, 1 ]),
					args=[ n, input.data_ptr(), flow.data_ptr(), gradOutput.data_ptr(), None, gradFlow.data_ptr() ]
				)
			# end

		elif input.is_cuda == False:
			raise NotImplementedError()

		# end

		return gradInput, gradFlow
	# end
# end

def FunctionSoftsplat(tenInput, tenFlow, tenMetric, strType):
	assert(tenMetric is None or tenMetric.shape[1] == 1)
	assert(strType in ['summation', 'average', 'linear', 'softmax'])

	if strType == 'average':
		tenInput = torch.cat([ tenInput, tenInput.new_ones(tenInput.shape[0], 1, tenInput.shape[2], tenInput.shape[3]) ], 1)

	elif strType == 'linear':
		tenInput = torch.cat([ tenInput * tenMetric, tenMetric ], 1)

	elif strType == 'softmax':
		tenInput = torch.cat([ tenInput * tenMetric.exp(), tenMetric.exp() ], 1)

	# end

	tenOutput = _FunctionSoftsplat.apply(tenInput, tenFlow)

	if strType != 'summation':
		tenOutput = tenOutput[:, :-1, :, :] / (tenOutput[:, -1:, :, :] + 0.0000001)
	# end

	return tenOutput
# end

class ModuleSoftsplat(torch.nn.Module):
	def __init__(self, strType):
		super(ModuleSoftsplat, self).__init__()

		self.strType = strType
	# end

	def forward(self, tenInput, tenFlow, tenMetric):
		return FunctionSoftsplat(tenInput, tenFlow, tenMetric, self.strType)
	# end
# end